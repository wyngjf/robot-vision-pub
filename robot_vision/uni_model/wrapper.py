import torch
import cv2
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from typing import Union, Literal, Tuple
from marshmallow_dataclass import dataclass
from rich.progress import track

from torchvision import transforms
from robot_utils import console
from robot_utils.py.utils import load_dataclass
from robot_utils.py.filesystem import validate_file
from robot_utils.cv.image.op_torch import get_imagenet_normalization
from robot_utils.cv.io.io_cv import load_rgb
from robot_utils.torch.torch_utils import get_device

from robot_vision.uni_model.uni_match import UniMatch
from robot_vision.uni_model.flow_viz import flow_to_image
from robot_vision.utils.utils import get_default_checkpoints_path


@dataclass
class UnimatchWrapperConfig:
    """
    The Mask detection wrapper, support grounded SAM or SAM-HQ
    Args:
        checkpoint: the abs path to the checkpoint file (.pth)
        flag_use_gpu: use torch.device gpu
        mode: the stereo depth estimation mode or optical flow mode
    """
    checkpoint:     str = None
    flag_use_gpu:   bool = True
    mode:           Literal['flow', 'stereo'] = "stereo"


class UnimatchWrapper:
    def __init__(
            self,
            cfg: Union[Path, dict, str, UnimatchWrapperConfig] = None
            # training_root,
            # device: torch.device,
            # mode: str = "flow",  # ["stereo", "flow"],
            # auto: bool = True,
            # search_pattern: List[str] = None,
    ):
        self.c = load_dataclass(UnimatchWrapperConfig, cfg)
        self.device = get_device(self.c.flag_use_gpu)

        norm_transform = get_imagenet_normalization()
        self.rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])
        self._load_model()

    def _reset_checkpoint(self):
        if self.c.checkpoint is None:
            checkpoint_root = get_default_checkpoints_path() / "unimatch"
            if self.c.mode == "stereo":
                self.c.checkpoint = checkpoint_root / "gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth"
            elif self.c.mode == "flow":
                self.c.checkpoint = checkpoint_root / "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
            else:
                raise NotImplementedError(f"{self.c.mode} is not supoorted")
            console.log(f"switching checkpoints to {self.c.checkpoint}")

        validate_file(self.c.checkpoint, throw_error=True)

    def _load_model(self):
        console.log(f"[bold blue]loading uminatch model")
        model_config = dict(
            task=self.c.mode,
            upsample_factor=4,
            num_scales=2,
            reg_refine=True,
            padding_factor=32,
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6
        )
        self.model = UniMatch(model_config).to(self.device)
        self.model.eval()
        self._load_checkpoint()

    def _load_checkpoint(self):
        self._reset_checkpoint()
        checkpoint = torch.load(str(self.c.checkpoint), map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)

    def change_mode(self, mode: Literal["stereo", "flow", "depth"], checkpoint: Union[str, Path] = None):
        self.c.checkpoint = checkpoint
        self.c.mode = mode
        self.model.change_mode(mode)
        self._load_checkpoint()
        self.model.to(self.device)

    def initialize(self, image: np.ndarray):
        self.model.get_image_info(self.rgb_image_to_tensor(image))

    def estimate_depth(self, img_l, img_r, camera_base_line, focal_length_w, depth_unit_in_meter=True):
        img_l = self.rgb_image_to_tensor(img_l).float()
        img_r = self.rgb_image_to_tensor(img_r).float()

        with torch.no_grad():
            depth_mm = self.model.get_depth(
                img_l.to(self.device).unsqueeze(0), img_r.to(self.device).unsqueeze(0),
                camera_base_line, focal_length_w
            )  # np.ndarray [[B, H*W, H*W], [B, H*W, H*W]]
            depth_mm = depth_mm.detach().cpu().numpy()
            if depth_unit_in_meter:
                return depth_mm * 0.001
            return depth_mm

    def get_colored_depth(self, depth_in_meter: np.ndarray):
        depth_image = cv2.normalize(depth_in_meter, None, 0, 1, cv2.NORM_MINMAX)
        colormap_image = cv2.applyColorMap(np.uint8(depth_image * 255), cv2.COLORMAP_JET)
        return colormap_image

    def estimate_flow(self, img_l, img_r) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the descriptors for image 1 and image 2 for each network
        """
        rgb_1_tensor = self.rgb_image_to_tensor(img_l / 255.).float().unsqueeze(0).to(self.device)
        rgb_2_tensor = self.rgb_image_to_tensor(img_r / 255.).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            results = self.model.forward_flow(rgb_1_tensor, rgb_2_tensor)  # [[B, H*W, H*W], [B, H*W, H*W]]

        flow_pr = results["flow_preds"][0]  # [B, 2, H, W]
        # resize back
        if self.model.resize_img:
            # console.log(f"size: {flow_pr.shape}, {self.model.raw_size}")
            flow_pr = F.interpolate(flow_pr, size=self.model.raw_size, mode='bilinear', align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * self.model.raw_size[-1] / self.model.inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * self.model.raw_size[-2] / self.model.inference_size[-2]

        if self.model.transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = np.round(flow_pr[0].permute(1, 2, 0).cpu().numpy()).astype(int)  # [H, W, 2]
        flow_img = flow_to_image(flow)
        return flow, flow_img


def estimate_depth_stereo(
        depth_wrapper: UnimatchWrapper,
        filenames_l: list,
        filenames_r: list,
        camera_base_line,
        focal_length_w,
        to_meter: bool = True
):
    """
    output: depth_list = [
        depth_in_mm_of_{idx}_th_image_pair: np.ndarray for idx in range(len(filenames_l))
    ]
    """
    scale = 0.001 if to_meter else 1.0
    if not len(filenames_l) == len(filenames_r):
        console.log("number of images for left / right views don't match")
        exit(1)
    console.log(f"In total {len(filenames_l)} image pairs")

    ref_img = load_rgb(filenames_l[0])
    depth_wrapper.initialize(ref_img)
    depth_list = []
    for idx in track(range(len(filenames_l)), description="[blue]Stereo depth estimation", console=console):
        img_l = load_rgb(filenames_l[idx])
        img_r = load_rgb(filenames_r[idx])
        depth_mm = depth_wrapper.estimate_depth(img_l, img_r, camera_base_line, focal_length_w, depth_unit_in_meter=False)
        depth_list.append(depth_mm * scale)
    return depth_list

