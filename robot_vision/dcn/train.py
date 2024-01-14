import click
from marshmallow_dataclass import dataclass
from typing import Union
from pathlib import Path
from rich.console import Group
from rich.progress import Progress, TextColumn
from rich.live import Live
import torch
import tensorboard_logger

from robot_utils import console
from robot_utils.py.utils import save_to_yaml, load_dict_from_yaml, load_dataclass_from_dict
from robot_utils.py.filesystem import create_path, get_ordered_files, copy2
from robot_utils.torch.torch_utils import init_torch

from robot_vision.utils.utils import get_dcn_path
from robot_vision.dcn.loss import HeatmapLoss, is_zero_loss
from robot_vision.dcn.model import DenseCorrespondenceNetwork
from robot_vision.dcn.vit_model import UniModel
from robot_vision.dcn.dataset import PDCDataset


@dataclass
class DCNTrainConfig:
    batch_size: int = 1
    num_workers: int = 1
    num_iterations: int = 20000
    train_config: str = None
    data_config: str = None
    use_gpu: bool = True

    save_rate: int = 200
    learning_rate: float = 1e-4
    learning_rate_decay: float = 0.9
    steps_for_decay: int = 250  # decay the learning rate after this many steps
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0

    enable_test: bool = False
    compute_test_loss_rate: int = 500  # how often to compute the test loss
    test_loss_num_iterations: int = 50  # how many samples to use to compute the test loss
    garbage_collect_rate: int = 1

    enable_vit: bool = False


class DenseCorrespondenceTraining:
    def __init__(self, path: Union[str, Path]):
        """
        Args:
            path: absolute path the training folder, should contain a data_config.yaml
        """
        self.root_path = Path(path)
        if not self.root_path.exists():
            console.print(f"[bold red]{path} does not exit")
            exit(1)

        train_config_file = self.root_path / "training.yaml"
        if not train_config_file.exists():
            src = get_dcn_path() / "config/training.yaml"
            copy2(src, train_config_file)

        cfg = load_dict_from_yaml(train_config_file)
        console.log(f"use config: {train_config_file}")

        self.c = load_dataclass_from_dict(DCNTrainConfig, cfg["train"])
        self.device = init_torch(use_gpu=self.c.use_gpu)

        self._load_dataset()
        self._loss = HeatmapLoss(cfg["loss"], device=self.device)
        self._loss.c.width = cfg['dcn']["image_width"] = self._dataset.w
        self._loss.c.height = cfg['dcn']["image_height"] = self._dataset.h

        if self.c.enable_vit:
            self._dcn = UniModel(cfg['dcn'], str(self.root_path)).to(self.device)
        else:
            self._dcn = DenseCorrespondenceNetwork(cfg['dcn'], str(self.root_path)).to(self.device)
        save_to_yaml(cfg, train_config_file, default_flow_style=False)
        self._construct_optimizer(self._dcn.parameters())
        self._setup_logging()

    def _load_dataset(self):
        data_config = Path(self.c.data_config) if self.c.data_config else self.root_path / "data_config.yaml"
        console.print(f"[green]Loading data from {data_config}")
        self._dataset = PDCDataset(config=data_config)
        self._data_loader = torch.utils.data.DataLoader(
            self._dataset, batch_size=self.c.batch_size, shuffle=True, num_workers=self.c.num_workers,
            pin_memory=True, drop_last=True
        )
        self.uv_flatten = self._dataset.get_uv_all(self.device)
        console.log(f"uv flatten: {self.uv_flatten.shape}")

        if self.c.enable_test:
            console.print("Load test dataset")
            self._dataset_test = PDCDataset(data_config, mode="eval")
            self._data_loader_test = torch.utils.data.DataLoader(
                self._dataset_test, batch_size=self.c.batch_size, shuffle=True, num_workers=self.c.num_workers,
                pin_memory=True, drop_last=True
            )

    def _setup_logging(self):
        self._logging_dir = create_path(self.root_path / "log")
        self._tensorboard_logger = tensorboard_logger.Logger(
            create_path(self._logging_dir / "tensorboard", remove_existing=False)
        )

        console.print(f"[bold green]logging dir: {self._logging_dir}\nsee also \n\n'tensorboard' subdir, or run \n")
        console.print(f"[bold cyan]tensorboard --logdir {self.root_path.parent}\n\n")

        self._logging_dict = dict(
            train={"iteration": [],
                   "loss": [],
                   "match_loss": [],
                   "learning_rate": [],
                   },
            test={"iteration": [], "loss": [], "match_loss": [], "non_match_loss": []}
        )

    def _construct_optimizer(self, parameters):
        self._optimizer = torch.optim.Adam(parameters, lr=self.c.learning_rate, weight_decay=self.c.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=self.c.steps_for_decay, gamma=self.c.learning_rate_decay)

    def _load_pretrained(self, model_folder: Union[str, Path], iteration: int = None, learning_rate: float = None):
        """
        Loads network and optimizer parameters from a previous training run.
        Args:
            model_folder: absolute path to training log directory
            iteration: force to start from 'iteration'. None to search in the model_folder and use the last checkpoint.

        Returns: the iteration to start
        """
        model_folder = Path(model_folder)
        if not model_folder.is_dir():
            console.print(f"[bold red]{model_folder} is not an absolute path")
            exit(1)

        if iteration is None:
            iteration = 0
            model_param_file = get_ordered_files(model_folder, pattern=[".pth"])[-1]
            optim_param_file = get_ordered_files(model_folder, pattern=[".opt"])[-1]
        else:
            model_param_file = model_folder / f"{iteration:>06d}.pth"
            optim_param_file = model_folder / f"{iteration:>06d}.opt"

        console.print(f"[green]load model from: {model_param_file}")

        self._dcn.load_state_dict(torch.load(str(model_param_file)))
        self._optimizer.load_state_dict(torch.load(str(optim_param_file)))
        if learning_rate is not None:
            self._optimizer.param_groups[0]['lr'] = learning_rate

        return iteration

    def run_from_pretrained(self, model_folder, iteration=None, learning_rate=None):
        iteration = self._load_pretrained(model_folder, iteration, learning_rate)
        self.run(start_iteration=iteration)

    def run(self, start_iteration=0):
        self._dcn.train()

        max_num_iterations = self.c.num_iterations + start_iteration
        p = Progress(console=console)
        log = Progress(TextColumn("{task.description}"), console=console)
        progress_group = Group(p, log)
        with Live(progress_group):
            console.rule("[bold blue]training Dense Correspondence Network")
            task = p.add_task("[blue]Training", total=self.c.num_iterations)
            log_task = log.add_task("", total=self.c.num_iterations)
            for epoch in range(1000):
                self._dataset.set_progress(start_iteration / max_num_iterations)
                for i, data in enumerate(self._data_loader, 0):
                    if isinstance(data, torch.Tensor) and data.nelement() == 0:
                        console.print("[red]empty data, continuing")
                        continue

                    start_iteration += 1
                    p.update(task, advance=1)

                    self._optimizer.zero_grad()
                    img_a, img_b, matches_a, matches_b = data

                    img_a, img_b = img_a.cuda(), img_b.cuda()
                    matches_a, matches_b = matches_a.cuda().squeeze(0), matches_b.cuda().squeeze(0)

                    # run both images through the network
                    if self.c.enable_vit:
                        descriptor_a, descriptor_b = self._dcn.forward_descriptor(img_a, img_b)
                        loss = self._loss.get_vit_loss(
                            descriptor_a, descriptor_b, matches_a, matches_b, start_iteration
                        )
                    else:
                        image_pred = self._dcn.forward_descriptor_image(torch.cat((img_a, img_b), dim=0))
                        descriptor_a, descriptor_b = image_pred[0], image_pred[1]
                        loss = self._loss.get(
                            descriptor_a, descriptor_b, matches_a, matches_b, start_iteration, self.uv_flatten
                        )

                    if is_zero_loss(loss):
                        break

                    if torch.isnan(loss):
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._dcn.parameters(), self.c.grad_clip)
                    self._optimizer.step()

                    # self.scheduler.step(loss.item())
                    self.scheduler.step()
                    log_str = f"loss: {loss.item():>10.7f} lr: {self._optimizer.param_groups[0]['lr']:>10.7f}"
                    log.update(log_task, description=log_str)

                    self.update_plots(start_iteration, loss)

                    if start_iteration % self.c.save_rate == 0 and start_iteration > 500:
                        self._dcn.save_model(optimizer=self._optimizer, start_iteration=start_iteration)

                    if start_iteration > max_num_iterations:
                        console.rule(f"[blue]Finished testing after {max_num_iterations} iterations")
                        self._dcn.save_model(optimizer=self._optimizer, start_iteration=start_iteration)
                        p.stop()
                        log.stop()
                        return

    def update_plots(self, start_iteration, loss, match_loss=None, masked_non_match_loss=None, background_non_match_loss=None, blind_non_match_loss=None):
        """
        Updates the tensorboard plots with current loss function information
        """

        learning_rate = self._optimizer.param_groups[0]['lr']
        self._logging_dict['train']['learning_rate'].append(learning_rate)
        self._tensorboard_logger.log_value("learning rate", learning_rate, start_iteration)

        #! Don't update any plots if the entry corresponding to that term is a zero loss
        if not is_zero_loss(match_loss):
            self._logging_dict['train']['match_loss'].append(match_loss.item())
            self._tensorboard_logger.log_value("train match loss", match_loss.item(), start_iteration)
        if not is_zero_loss(masked_non_match_loss):
            self._logging_dict['train']['masked_non_match_loss'].append(masked_non_match_loss.item())
            self._tensorboard_logger.log_value("train masked non match loss", masked_non_match_loss.item(), start_iteration)
        if not is_zero_loss(background_non_match_loss):
            self._logging_dict['train']['background_non_match_loss'].append(background_non_match_loss.item())
            self._tensorboard_logger.log_value("train background non match loss", background_non_match_loss.item(), start_iteration)
        if not is_zero_loss(blind_non_match_loss):
            self._tensorboard_logger.log_value("train blind SINGLE_OBJECT_WITHIN_SCENE", blind_non_match_loss.item(), start_iteration)

        self._tensorboard_logger.log_value("train loss SINGLE_OBJECT_WITHIN_SCENE", loss.item(), start_iteration)

    def save_network(self, dcn, optimizer, iteration):
        torch.save(dcn.state_dict(), str(self._logging_dir / f"{iteration:>06d}.pth"))
        torch.save(optimizer.state_dict(), str(self._logging_dir / f"{iteration:>06d}.opt"))


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",    "-p",   type=str,       help="the absolute path to the training root folder")
def main(path):
    train = DenseCorrespondenceTraining(path)
    train.run()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
