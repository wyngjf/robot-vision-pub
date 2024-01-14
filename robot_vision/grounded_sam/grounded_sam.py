import cv2
import click
import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
from marshmallow_dataclass import dataclass

from robot_utils import console
from robot_utils.py.utils import load_dataclass
from robot_utils.py.filesystem import create_path
from robot_utils.cv.io.io_cv import load_rgb
from robot_utils.cv.opencv import overlay_masks_on_image
from robot_utils.torch.torch_utils import get_device

import robot_vision.grounded_sam.grouding_dino as gd
from robot_vision.grounded_sam.sam import SAMConfig, build_model
from robot_vision.utils.utils import get_root_path


@dataclass
class GroundedSAMConfig:
    """
    Grounded SAM configuration
    Args:
        sam: SAM configuration
        llm: Grounding DINO configuration
        flag_use_gpu: to specify the torch device
        flag_highest_score_only: whether to only detect masks with highest score
    """
    sam: SAMConfig = SAMConfig()
    llm: gd.GroundingDINOConfig = gd.GroundingDINOConfig()
    flag_use_gpu: bool = True
    flag_highest_score_only: bool = False


class GroundedSAM:
    def __init__(self, config: Union[str, Path, Dict, GroundedSAMConfig] = None):
        """
        Grounded SAM model. You don't have to set the text_prompt in llm config
         when initializing the model, but then you have to provide it when calling
         the run() method

        Args:
            config: configuration for GroundedSAM
        """
        self.c = load_dataclass(GroundedSAMConfig, config)
        self.device = get_device(self.c.flag_use_gpu)

        self.det_model = gd.build_model(config=self.c.llm).to(self.device)
        self.sam_model = build_model(config=self.c.sam)
        self.sam_model.model = self.sam_model.model.to(self.device)

    def run(self, image: np.ndarray, text_prompt: str) -> Dict[str, Any]:
        """
        the output tensor is represented in torch.Tensor
        Args:
            image:
            text_prompt:

        Returns: return empty dictionary if no object is detected.

        """
        if not text_prompt:
            if not self.c.llm.text_prompt:
                console.log(f"[bold red]your text prompt is invalid")
                return {}
        else:
            self.c.llm.text_prompt = text_prompt

        det_model, pred_dict = gd.run(self.det_model, image, self.c.llm, device=self.device)

        if pred_dict['boxes'].shape[0] == 0:
            console.log('[red bold]No objects detected!')
            return {}

        self.sam_model.set_image(image)

        transformed_boxes = self.sam_model.transform.apply_boxes_torch(
            pred_dict['boxes'], image.shape[:2])
        transformed_boxes = transformed_boxes.to(self.device)

        masks, _, _ = self.sam_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            # hq_token_only=self.c.sam.flag_sam_hq
        )
        pred_dict['masks'] = masks

        if self.c.flag_highest_score_only:
            labels = pred_dict['labels']
            scores = pred_dict['scores']
            scores_np = np.array([float(s) for s in pred_dict['scores']])
            label_set = list(set(pred_dict['labels']))
            idx_set = []
            for label in label_set:
                idx = [i for i, _label in enumerate(pred_dict['labels']) if _label == label]
                idx_set.append(idx[np.argmax(scores_np[idx])])
            pred_dict['labels'] = label_set
            pred_dict['scores'] = [pred_dict['scores'][i] for i in idx_set]
            pred_dict['boxes'] = pred_dict['boxes'][idx_set]
            if 'masks' in pred_dict:
                pred_dict['masks'] = pred_dict['masks'][idx_set]

        return pred_dict


def visualize_pred(
        image,
        pred_dict,
        show_label=True,
):
    with_mask = 'masks' in pred_dict
    labels = pred_dict['labels']
    scores = pred_dict['scores']

    bboxes = pred_dict['boxes'].cpu().numpy()

    for box, label, score in zip(bboxes, labels, scores):
        box = np.round(box).astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        if show_label:
            text = f'{label}|{score}' if isinstance(score, str) else f'{label}|{round(score,2)}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(image, text, (box[0], box[1]), font, fontScale, color, thickness, cv2.LINE_AA)

    if with_mask:
        masks = pred_dict['masks']
        if isinstance(masks, torch.Tensor):
            masks = masks.squeeze(1).cpu().numpy()
        image = overlay_masks_on_image(image, masks)

    return image


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--image_path",       "-p",   type=str,       help="the absolute path of the input image")
@click.option("--text_prompt",      "-t",   type=str,       help="stereo or mono")
@click.option("--high_score_only",  "-h",   is_flag=True,   help="only detect highest score")
@click.option("--hq_token_only",    "-hq",  is_flag=True,   help="use the high-quality SAM")
def main(image_path: str, text_prompt: str, high_score_only: bool, hq_token_only: bool):
    config = GroundedSAMConfig()
    config.flag_highest_score_only = high_score_only
    config.sam.flag_sam_hq = hq_token_only
    model = GroundedSAM(config)

    image_path = Path(image_path)
    image = load_rgb(image_path, bgr2rgb=True)
    out_dir = create_path(get_root_path().parent / "output")
    save_path = out_dir / image_path.name
    ic(save_path)

    pred_dict = model.run(image, text_prompt)
    image = visualize_pred(image, pred_dict)
    cv2.imwrite(str(save_path), image[:, :, ::-1])


if __name__ == "__main__":
    main()
