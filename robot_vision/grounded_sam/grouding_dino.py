import torch
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Union, Dict
from mmengine.config import Config
from marshmallow_dataclass import dataclass

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model as build_grounding_dino_model
from groundingdino.util import get_tokenlizer
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from robot_vision.utils.utils import get_root_path, get_default_checkpoints_path

from robot_utils import console
from robot_utils.torch.torch_utils import get_device
from robot_utils.py.filesystem import validate_file
from robot_utils.py.utils import load_dataclass


grounding_dino_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@dataclass
class GroundingDINOConfig:
    """
    Grounding DINO configuration
    Args:
        checkpoints: the full path to the (.pth) file
    """
    text_prompt:                    str = None
    checkpoints:                    str = None
    config_file:                    str = None
    apply_original_groudingdino:    bool = False
    box_thr:                        float = 0.22
    text_thr:                       float = 0.25
    highest_score_only:             bool = False


def build_model(
        config: Union[str, Path, Dict, GroundingDINOConfig] = None,
):
    config = load_dataclass(GroundingDINOConfig, config)
    if not config.config_file or not validate_file(config.config_file)[1]:
        config.config_file = get_root_path() / "grounded_sam/configs/GroundingDINO_SwinT_OGC.py"
        console.log(f"[bold red]using default configuration file: {config.config_file}")
    validate_file(config.config_file, throw_error=True)

    if not config.checkpoints or not validate_file(config.checkpoints)[1]:
        config.checkpoints = get_default_checkpoints_path() / "groundingdino/groundingdino_swint_ogc.pth"

    gdino_args = Config.fromfile(str(config.config_file))
    model = build_grounding_dino_model(gdino_args)
    checkpoint = torch.load(str(config.checkpoints), map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j,
    if token i is mapped to j label"""

    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = []
            for i in range(beg_pos, end_pos + 1):
                positive_map_label_to_token[labels[j]].append(i)

    return positive_map_label_to_token


def convert_grounding_to_od_logits(
        logits,
        num_classes,
        positive_map,
        score_agg='MEAN'
):
    """
    logits: (num_query, max_seq_len)
    num_classes: 80 for COCO
    """
    assert logits.ndim == 2
    assert positive_map is not None
    scores = torch.zeros(logits.shape[0], num_classes).to(logits.device)
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == 'MEAN':  # True
        for label_j in positive_map:
            scores[:, label_j] = logits[:, torch.LongTensor(positive_map[label_j])].mean(-1)
    else:
        raise NotImplementedError
    return scores


def run(model, image: np.ndarray, c: GroundingDINOConfig, device=None):
    if not c.text_prompt:
        raise ValueError(f"no prompts is specified")
    if device is None:
        device = get_device(use_gpu=True)
    pred_dict = {}

    image_pil = Image.fromarray(image)
    image_pil = apply_exif_orientation(image_pil)  # type: Image
    image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

    text_prompt = c.text_prompt
    text_prompt = text_prompt.lower()
    text_prompt = text_prompt.strip()
    if not text_prompt.endswith('.'):
        text_prompt = text_prompt + '.'

    # Original GroundingDINO use text-thr to get class name,
    # the result will always result in categories that we don't want,
    # so we provide a category-restricted approach to address this

    if not c.apply_original_groudingdino:
        # custom label name
        custom_vocabulary = text_prompt[:-1].split('.')
        label_name = [phrase.strip() for phrase in custom_vocabulary]
        tokens_positive = []
        start_i = 0
        separation_tokens = ' . '
        for _index, label in enumerate(label_name):
            end_i = start_i + len(label)
            tokens_positive.append([(start_i, end_i)])
            if _index != len(label_name) - 1:
                start_i = end_i + len(separation_tokens)
        tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
        tokenized = tokenizer(c.text_prompt, padding='longest', return_tensors='pt')
        positive_map_label_to_token = create_positive_dict(
            tokenized, tokens_positive, list(range(len(label_name))))

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image.unsqueeze(0), captions=[text_prompt])

    logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

    if not c.apply_original_groudingdino:
        logits = convert_grounding_to_od_logits(
            logits, len(label_name), positive_map_label_to_token)  # [N, num_classes]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > c.box_thr
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    if c.apply_original_groudingdino:
        # get phrase
        tokenizer = model.tokenizer
        tokenized = tokenizer(text_prompt)
        # build pred
        pred_labels = []
        pred_scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > c.text_thr, tokenized, tokenizer)
            pred_labels.append(pred_phrase)
            pred_scores.append(str(logit.max().item())[:4])
    else:
        scores, pred_phrase_idxs = logits_filt.max(1)
        # build pred
        pred_labels = []
        pred_scores = []
        for score, pred_phrase_idx in zip(scores, pred_phrase_idxs):
            pred_labels.append(label_name[pred_phrase_idx])
            pred_scores.append(str(score.item())[:4])

    pred_dict['labels'] = pred_labels
    pred_dict['scores'] = pred_scores

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    pred_dict['boxes'] = boxes_filt

    return model, pred_dict


def apply_exif_orientation(image: Image):
    """Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`.
    The Pillow source raises errors with
    various methods, especially `tobytes`
    Function based on:
      https://github.com/facebookresearch/detectron2/\
      blob/78d5b4f335005091fe0364ce4775d711ec93566e/\
      detectron2/data/detection_utils.py#L119
    Args:
        image (PIL.Image): a PIL image
    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    _EXIF_ORIENT = 274
    if not hasattr(image, 'getexif'):
        return image

    try:
        exif = image.getexif()
    except Exception:
        # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)
    if method is not None:
        return image.transpose(method)
    return image
