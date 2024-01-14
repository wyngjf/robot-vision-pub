from pathlib import Path
from typing import Union, List, Dict, Any
from dataclasses import dataclass
import torch
import cv2
import numpy as np

# from torchvision import transforms
from robot_utils import console
from robot_utils.py.filesystem import validate_path, validate_file, get_ordered_subdirs
from robot_utils.py.utils import load_dict_from_yaml, save_to_yaml
from robot_utils.py.interact import ask_list
from robot_utils.cv.image.op_torch import ResizedCrop
from robot_utils.cv.io.io_cv import load_rgb
from robot_utils.torch.torch_utils import get_device

from robot_vision.dcn.dataset import PDCDataset
from robot_vision.dcn.model import DenseCorrespondenceNetwork
from robot_vision.utils.utils import get_default_checkpoints_path


def update_config(dcn_training_root: Union[str, Path], overwrite: bool = False) -> dict:
    dcn_path, b = validate_path(Path(dcn_training_root), throw_error=True)
    config_filename = dcn_path / "config.yaml"
    if not validate_file(config_filename)[1]:
        config = dict(
            all={},
            user={},
        )
    else:
        config = load_dict_from_yaml(config_filename)

    all_obj = get_ordered_subdirs(dcn_path)
    for obj in all_obj:
        obj_name = obj.stem
        snapshots = get_ordered_subdirs(obj)
        c = config["all"].get(obj_name, {})
        user_c = config["user"].get(obj_name, "")
        for snapshot in snapshots:
            data_config_file, b = validate_file(snapshot / "data_config.yaml")
            note = "" if not b else load_dict_from_yaml(data_config_file).get("note")
            c[snapshot.name] = note
        config["all"][obj_name] = c
        config["user"][obj_name] = user_c
    if overwrite:
        save_to_yaml(config, config_filename, default_flow_style=False)
    return config


@dataclass
class DCNData:
    obj_name: str = ""
    path: Path = None
    model: DenseCorrespondenceNetwork = None
    dataset: PDCDataset = None
    dim: int = None
    image_wh: tuple = None
    intrinsics: np.ndarray = None
    scale: float = 1.0


class DCNWrapper:
    def __init__(
            self,
            object_dict: Dict[str, str],
            training_root: Union[str, Path] = None,
            device: torch.device = None,
            load_dataset: bool = False,
            mode: str = "eval",
            auto_select: bool = True,
    ):
        """
        This is a wrapper of DCN model used for inference.

        the folders of the training_root looks like the follows:
        training_root
        - dcn
            - config.yaml
            - object1
                - snapshot1
                - snapshot2
                - ...
            - object2
        """
        from robot_utils.py.interact import user_warning
        user_warning("The args of construction changed from obj list to obj dict")
        if not validate_path(training_root)[1]:
            training_root = get_default_checkpoints_path() / "dcn"
        self.dcn_path = validate_path(training_root, throw_error=True)[0]
        object_list = object_dict.keys()
        self.obj_dict = object_dict
        self.obj_checkpoint_name_list = [object_dict[obj] for obj in object_list]
        self.obj_list = object_list
        self.device = get_device(True) if device is None else device
        self._load_dcn(load_dataset, mode, auto_select)

        self.transform = ResizedCrop(with_to_tensor=True, with_imagenet_norm=True)

    def _load_dcn(
            self,
            load_dataset: bool = False,
            mode: str = "eval",
            auto_select: bool = True,
    ):
        config = update_config(self.dcn_path)
        self.root_path_dcn = {}
        self.dcn = {}
        for i, (obj_name, obj_cp_name) in enumerate(zip(self.obj_list, self.obj_checkpoint_name_list)):
            if auto_select:
                if not config["user"][obj_cp_name]:
                    raise ValueError(f"dcn model for {obj_name} ({obj_cp_name}) is "
                                     f"not specified in {self.dcn_path}/config.yaml")
                dcn_path = self.dcn_path / obj_cp_name / config["user"][obj_cp_name]
            else:
                dcn_path = ask_list("select one DCN model", get_ordered_subdirs(self.dcn_path / obj_cp_name))

            dcn_path, b = validate_path(dcn_path)
            if not b:
                console.log(f"[bold red]DCN Path {dcn_path} doesn't exist for object {obj_cp_name}")
                exit(1)

            cfg = load_dict_from_yaml(dcn_path / "training.yaml")['dcn']
            data_cfg = load_dict_from_yaml(dcn_path / "data_config.yaml")
            model = DenseCorrespondenceNetwork(cfg, str(dcn_path)).to(self.device)
            model.load_model()
            console.log(f"DCN checkpoint: {model.params_file}")
            model.eval()
            self.dcn[obj_name] = DCNData(
                obj_name=obj_cp_name, path=dcn_path, model=model,
                dim=model.c.descriptor_dimension, image_wh=(model.c.image_width, model.c.image_height),
                intrinsics=np.array(data_cfg["cam_intrinsic"]),
                scale=data_cfg.get("scale", 1.0)
            )
            if load_dataset:
                self.dcn[obj_name].dataset = PDCDataset(dcn_path / "data_config.yaml", mode=mode)

    def compute_descriptor_images(
            self,
            images: Union[np.ndarray, List[np.ndarray], List[Path], List[str]],
            descriptors: Dict[str, Any] = None
    ) -> Dict[str, torch.Tensor]:
        if isinstance(images, np.ndarray):
            images = [images] * len(self.obj_list)
        elif isinstance(images, list):
            if isinstance(images[0], Path) or isinstance(images[0], str):
                img = []
                for f in images:
                    f, _ = validate_file(f, throw_error=True)
                    img.append(load_rgb(f))
                images = img
        if len(images) != len(self.obj_list):
            console.log(f"[bold red]the number of images doesn't match the number of objects")
            exit(1)
        if descriptors is None:
            descriptors = {}
        for obj_name, img in zip(self.obj_list, images):
            original_size = img.shape[:2]
            image_rgb = cv2.resize(img, self.dcn[obj_name].image_wh)
            _rgb_tensor = self.transform.forward(image_rgb)
            descriptor_image = self.dcn[obj_name].model.forward_single_image_tensor(
                _rgb_tensor, resize_hw=original_size
            )
            descriptors[obj_name] = descriptor_image.detach().cpu()
        return descriptors


if __name__ == "__main__":
    p = "/media/gao/temp/fromJeff/kvil_process/train/dcn"
    update_config(p)
