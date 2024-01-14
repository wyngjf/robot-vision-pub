import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
from marshmallow_dataclass import dataclass

from robot_utils import console
from robot_utils.py.filesystem import validate_path
from robot_utils.py.utils import load_dataclass

from robot_vision.grounded_sam.grounded_sam import GroundedSAM, GroundedSAMConfig
from robot_vision.grounded_sam.utils import get_indices_of_item_in_list


@dataclass
class MaskWrapperConfig(GroundedSAMConfig):
    """
    The Mask detection wrapper, support grounded SAM or SAM-HQ
    Args:
        training_root: the root path containing checkpoints for different models, e.g. training_root/sam/xxx.pth
        obj_prompt_cfg: the list of obj names, or dict of map from obj name to their prompt description
    """
    training_root: str = None
    obj_prompt_cfg: Union[List[str], Dict[str, str]] = None


class MaskWrapper:
    def __init__(
            self,
            cfg: Union[Path, dict, str, MaskWrapperConfig] = None
    ):
        """
        Instantiate Mask detection wrapper, support grounded SAM or SAM-HQ

        Args:
            cfg: the cfg filename or the cfg object
        """
        console.rule("[bold cyan]loading Grounded SAM as segmentation model")
        self.c = load_dataclass(MaskWrapperConfig, cfg)

        self.obj_dict: Optional[Dict[str, str]] = None
        self.text_prompt = None

        # self.mask_path = validate_path(Path(self.c.training_root), throw_error=True)[0]
        self._load_model()

    def reset_obj_prompt(
            self,
            obj_prompt_cfg: Union[str, List[str], Dict[str, str]]
    ):
        if isinstance(obj_prompt_cfg, str):
            self.obj_dict = {obj_prompt_cfg: obj_prompt_cfg}
        elif isinstance(obj_prompt_cfg, list):
            self.obj_dict = {o: o for o in obj_prompt_cfg}
        elif isinstance(obj_prompt_cfg, dict):
            self.obj_dict = obj_prompt_cfg
        else:
            raise TypeError(f"the type of obj_prompt_cfg {type(obj_prompt_cfg)} is not supported, use "
                            f"str, List[str] or Dict[str, str] instead")

        self.text_prompt = " . ".join([v for k, v in self.obj_dict.items()])

    def _load_model(self):
        # console.rule(f"[bold blue]loading Grounded SAM from {self.mask_path}")
        self.model = GroundedSAM(self.c)
        self.device = self.model.device

    def get_masks(
            self,
            image:              np.ndarray,
            erode_radius:       Union[int, Dict[str, int]] = -1,
            obj_cfg:            Union[str, List[str], Dict[str, str]] = None,
            expand_dim:         int = -1,
            del_torch_tensor:   bool = True,
            as_binary:          bool = False,
    ) -> Union[Dict[str, List[np.ndarray]], None]:
        self.reset_obj_prompt(obj_cfg)

        pred_dict = self.model.run(image, self.text_prompt)
        if not pred_dict:
            console.log(f"[bold red]No mask is detected")
            return dict.fromkeys(self.obj_dict.keys())

        pred_dict["masks"] = pred_dict["masks"].squeeze(1)  # (n_masks, 1, h, w) -> (n_masks, h, w)
        pred_dict["masks_np"] = {}
        masks = pred_dict["masks"].cpu().numpy()  # (n_masks, 1, h, w) -> (n_masks, h, w)

        for o in self.obj_dict.keys():
            obj_idx = get_indices_of_item_in_list(pred_dict["labels"], self.obj_dict[o])
            if len(obj_idx) == 0:
                console.log(f"[red]No mask is detected for object: {o}")
                pred_dict["masks_np"][o] = []
                continue

            kernel = None
            if isinstance(erode_radius, int) and erode_radius >= 1:
                kernel = np.ones((erode_radius, erode_radius), np.uint8)
            elif isinstance(erode_radius, dict) and erode_radius[o] >= 1:
                _erode_radius = erode_radius[o]
                kernel = np.ones((_erode_radius, _erode_radius), np.uint8)

            if as_binary:
                obj_mask = [masks[i].astype(bool) for i in obj_idx]
            else:
                obj_mask = [masks[i].astype(np.uint8) for i in obj_idx]

            for i in range(len(obj_mask)):
                if kernel is not None:
                    obj_mask[i] = cv2.erode(obj_mask[i].astype(np.uint8), kernel, iterations=1)

                if isinstance(expand_dim, int) and expand_dim > 1:
                    obj_mask[i] = np.expand_dims(obj_mask[i], axis=2).repeat(expand_dim, axis=2)

            pred_dict["masks_np"][o] = obj_mask
        if del_torch_tensor:
            del pred_dict["masks"]
        return pred_dict

