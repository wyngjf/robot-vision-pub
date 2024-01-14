from pathlib import Path
from typing import Union, Literal, Dict, Tuple

import numpy as np
from marshmallow_dataclass import dataclass

from scipy import ndimage
import cv2
import copy
from robot_utils.cv.opencv import overlay_masks_on_image

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from robot_utils import console
from robot_utils.py.utils import load_dataclass
from robot_utils.py.filesystem import validate_file
from robot_vision.utils.utils import get_default_checkpoints_path


@dataclass
class SAMConfig:
    """
    SAM configuration

    Args:
        checkpoints: the full path to the (.pth) file
        sam_type: the SAM model type
        flag_sam_hq: whether to use SAM-HQ
    """
    checkpoints:                str = None
    sam_type:                   Literal['vit_h', 'vit_l', 'vit_b'] = "vit_h"
    flag_sam_hq:                bool = False
    th_mask_confidence_lower:   float = 0.1


def build_model(
        config: Union[str, Path, Dict, SAMConfig] = None,
) -> SamPredictor:
    """
    Build the SamPredictor model
    """
    config = load_dataclass(SAMConfig, config)

    if not config.checkpoints or not validate_file(config.checkpoints)[1]:
        sam_folder = "sam_hq" if config.flag_sam_hq else "sam"
        model_file = f"{sam_folder}/{sam_folder}_{config.sam_type}.pth"
        config.checkpoints = get_default_checkpoints_path() / model_file
        console.log(f"[bold red]Using default checkpoints {config.checkpoints}")

    build_sam = sam_model_registry[config.sam_type]
    return SamPredictor(build_sam(checkpoint=str(config.checkpoints)))


class SAMForAll:
    def __init__(
            self,
            config: Union[str, Path, Dict, SAMConfig] = None,
            device: str = "cuda:0"
    ):
        self.sam_model = build_model(config)
        self.sam_model.model = self.sam_model.model.to(device)

    def build_sam_to_segment_all(
            self,
            img: np.ndarray,                    # (h, w, 3)
            sampled_grid_wh: np.ndarray,        # (N, 2)
            pred_iou_thresh:    float = 0.9,
    ):
        mask_generator = SamAutomaticMaskGenerator(
            self.sam_model.model,
            crop_overlap_ratio=0.5,
            pred_iou_thresh=pred_iou_thresh,
            points_per_side=None,
            point_grids=[sampled_grid_wh / np.array([img.shape[:2]])[:, ::-1]]  # each point (w, h)
        )
        inner_obj_masks = mask_generator.generate(img)
        all_mask_overlay, obj_annotations = post_annotation(
            copy.deepcopy(img), inner_obj_masks, None, draw=True,
            area_thr=(0, 1000000000)
        )
        return all_mask_overlay


def post_annotation(image: np.ndarray, annotations, contour=None, draw: bool = False, area_thr: Tuple[int, int] = (1e0, 1e10)):
    """
    annotations is the segment_anything format, please check code
    https://github.com/facebookresearch/segment-anything/blob/567662b0fd33ca4b022d94d3b8de896628cd32dd/segment_anything/automatic_mask_generator.py#L144
    """
    if len(annotations) == 0:
        console.log("[bold red]no object detected")
        return
    annotations = sorted(annotations, key=(lambda x: x['area']), reverse=True)

    # for i in range(len(annotations)):
    #     console.log(f"[bold yellow]{annotations[i]}")

    # object_ids_to_suppress = []
    # for i, ann in enumerate(annotations):
    #     if ann["area"] < area_thr[0] or ann["area"] > area_thr[1]:
    #         console.log(f"[bold red]{ann['area']} not in range {area_thr}")
    #         object_ids_to_suppress.append(i)
    # for id in sorted(set(object_ids_to_suppress), reverse=True):
    #     console.log(f"[bold yellow]Suppressing {id}")
    #     del annotations[id]

    # # supress all masks that are not unique (remove the smaller objects)
    # object_ids_to_suppress = []
    # for i in range(len(annotations)):
    #     for j in range(i + 1, len(annotations)):
    #         inner_obj_mask_i = annotations[i]['segmentation']
    #         inner_obj_mask_j = annotations[j]['segmentation']
    #
    #         intersection = np.bitwise_and(inner_obj_mask_i, inner_obj_mask_j)
    #
    #         intersection_i = np.bitwise_and(intersection, inner_obj_mask_i)
    #         intersection_j = np.bitwise_and(intersection, inner_obj_mask_j)
    #
    #         # c_intersec = np.count_nonzero(intersection)
    #         c_i = np.count_nonzero(intersection_i)
    #         c_j = np.count_nonzero(intersection_j)
    #         r_i = np.count_nonzero(inner_obj_mask_i)
    #         r_j = np.count_nonzero(inner_obj_mask_j)
    #
    #         fraction_i = c_i / r_i
    #         fraction_j = c_j / r_j
    #
    #         if fraction_i > 0.95:
    #             object_ids_to_suppress.append(i)
    #         elif fraction_j > 0.95:
    #             object_ids_to_suppress.append(j)

    # for id in sorted(set(object_ids_to_suppress), reverse=True):
    #     console.log(f"[bold yellow]Suppressing {id}")
    #     del annotations[id]

    sorted_annotations = []
    for ann in annotations:
        bbox = ann["bbox"]
        com = ndimage.center_of_mass(ann["segmentation"])
        if contour is not None and cv2.pointPolygonTest(contour, (com[1], com[0]), False) < 0:
            console.log(f"[bold red]point {(round(bbox[1]+0.5 * bbox[3]), round(bbox[0]+0.5*bbox[2]))} out of contour")
            continue
        sorted_annotations.append(ann)

    if not draw:
        return image, sorted_annotations

    colors = [np.random.randint(0, 255, 3).astype(np.uint8) for _ in range(len(sorted_annotations))]

    for ann, c in zip(sorted_annotations, colors):
        m = ann['segmentation']
        box = ann["bbox"]
        box[2] += box[0]
        box[3] += box[1]
        image = overlay_masks_on_image(image, [m], [c])
        # ic(box)
        # image_shape = image.shape[:2]
        # box[0] = max(0, box[0])
        # box[1] = min(image_shape[1], box[1])
        # box[2] = max(0, box[2])
        # box[3] = min(image_shape[0], box[3])
        # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # predicted_iou = ann["predicted_iou"]
        # stability_score = ann["stability_score"]
        # text = f'{predicted_iou:>.2f}__{stability_score:>.2f}'
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale = 1
        # color = (255, 255, 255)
        # thickness = 2
        # cv2.putText(image, text, (box[0], box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
    return image, sorted_annotations



