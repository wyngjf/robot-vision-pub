import numpy as np
import torch
import torch.nn.functional as F

from robot_utils import console
from robot_vision.uni_model.flow_viz import save_vis_flow_tofile
from robot_vision.uni_model.uni_match import UniMatch


@torch.no_grad()
def inference_flow(
        model: UniMatch,
        image1: torch.Tensor,
        image2: torch.Tensor,
        output_file: str = "",
):
    """ Inference on a directory or a video """
    model.eval()
    results_dict = model.forward_flow(image1, image2)

    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

    # resize back
    if model.resize_img:
        flow_pr = F.interpolate(flow_pr, size=model.raw_size, mode='bilinear', align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * model.raw_size[-1] / model.inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * model.raw_size[-2] / model.inference_size[-2]

    if model.transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)

    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    save_vis_flow_tofile(flow, output_file)
    console.rule('Done!')
