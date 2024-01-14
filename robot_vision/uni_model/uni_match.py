"""
This module is modified based on the work https://github.com/LiheYoung/UniMatch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import transforms
# from torchvision.transforms.functional import hflip
from math import ceil
from pathlib import Path
from typing import List, Union, Literal
from marshmallow_dataclass import dataclass

from robot_utils import console
from robot_utils.py.utils import load_dataclass, default_field
from robot_utils.cv.image.op_torch import get_imagenet_normalization
from robot_utils.cv.io.io_torch import load_rgb
from robot_utils.torch.geometry import flow_warp, compute_flow_with_depth_pose
from robot_utils.cv.correspondence.matching import (
    global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow, global_prob,
    global_correlation_softmax_stereo, local_correlation_softmax_stereo, correlation_softmax_depth)

from robot_vision.cnn.res_encoder import CNNEncoder
from robot_vision.vit.transformer import FeatureTransformer
from robot_vision.vit.attention import SelfAttnPropagation
from robot_vision.uni_model.reg_refine import BasicUpdateBlock
from robot_vision.vit.utils import normalize_img, feature_add_position, upsample_flow_with_mask


@dataclass
class ModelConfig:
    # model: learnable parameters
    task:                   Literal['flow', 'stereo', 'depth'] = "flow"
    num_scales:             int = 1             # feature scales: 1/8 or 1/8 + 1/4')
    feature_channels:       int = 128
    upsample_factor:        int = 8
    num_head:               int = 1
    ffn_dim_expansion:      int = 4
    num_transformer_layers: int = 6
    reg_refine:             bool = False        # optional task-specific local regression refinement

    # model: parameter-free
    attn_type:              str = 'swin'        # attention function
    attn_splits_list:       List[int] = default_field([2])     # number of splits in attention
    corr_radius_list:       List[int] = default_field([-1])    # correlation radius for matching, -1 indicates global matching
    prop_radius_list:       List[int] = default_field([-1])    # self-attention radius for propagation, -1 indicates global attention
    num_reg_refine:         int = 1             # number of additional local regression refinement

    padding_factor:         int = 8

    max_depth_in_mm:        float = 5000.0

    # loss
    gamma:                  float = 0.9         # exponential weighting

    def __post_init__(self):
        if self.task != 'depth':
            assert len(self.attn_splits_list) == len(self.corr_radius_list) == len(
                self.prop_radius_list) == self.num_scales
        else:
            # Depth mode # not supported for multi-scale depth model
            assert len(self.attn_splits_list) == len(self.prop_radius_list) == self.num_scales == 1


class UniMatch(nn.Module):
    def __init__(self, config: Union[Path, dict]):
        super(UniMatch, self).__init__()
        self.c = load_dataclass(ModelConfig, config)
        self.feature_channels = self.c.feature_channels
        self.num_scales = self.c.num_scales
        self.upsample_factor = self.c.upsample_factor
        self.reg_refine = self.c.reg_refine

        self.transpose_img = False
        self.resize_img = False
        self.inference_size = None
        self.raw_size = None

        # CNN
        self.backbone = CNNEncoder(output_dim=self.c.feature_channels, num_output_scales=self.c.num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=self.c.num_transformer_layers,
                                              d_model=self.c.feature_channels,
                                              nhead=self.c.num_head,
                                              ffn_dim_expansion=self.c.ffn_dim_expansion,
                                              )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=self.c.feature_channels)
        self._reset(self.c.task)

    def _reset(self, task: str):
        if task not in ["stereo", "flow", "depth"]:
            raise ValueError(f"{task} must be one of the following 'flow', 'stereo', 'depth'")
        self.c.task = task

        if not self.reg_refine or self.c.task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(
                nn.Conv2d(2 + self.c.feature_channels, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.c.upsample_factor ** 2 * 9, 1, 1, 0)
            )
            # thus far, all the learnable parameters are task-agnostic
        else:
            self.upsampler = None

        if self.c.reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=self.c.upsample_factor,
                                           flow_dim=2 if self.c.task == 'flow' else 1,
                                           bilinear_up=self.c.task == 'depth',
                                           )

    def change_mode(self, task: Literal["stereo", "flow", "depth"]):
        self._reset(task)

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            # mask: [1, 144, 184, 320], flow: [1, 2, 184, 320], feature: [1, 128, 184, 320]
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def forward(self, img0, img1,
                pred_bidir_flow=False,
                intrinsics=None,
                pose=None,  # relative pose transform
                min_depth=1. / 0.5,  # inverse depth range
                max_depth=1. / 10,
                num_depth_candidates=64,
                depth_from_argmax=False,
                pred_bidir_depth=False,
                **kwargs,
                ):
        """
        all images should be normalized in the dataset
        """
        task = self.c.task
        if pred_bidir_flow:
            assert task == 'flow'

        if task == 'depth':
            assert self.num_scales == 1  # multi-scale depth model is not supported yet

        results_dict = {}
        flow_preds = []

        # if task == 'flow':
        #     # stereo and depth tasks have normalized img in dataloader
        #     img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        if task != 'depth':
            assert len(self.c.attn_splits_list) == len(self.c.corr_radius_list) == len(
                self.c.prop_radius_list) == self.num_scales
        else:
            assert len(self.c.attn_splits_list) == len(self.c.prop_radius_list) == self.num_scales == 1

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            feature0_ori, feature1_ori = feature0, feature1

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if task == 'depth':
                # scale intrinsics
                intrinsics_curr = intrinsics.clone()
                intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / upsample_factor

            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                assert task != 'depth'
                flow = flow.detach()

                if task == 'stereo':
                    # construct flow vector for disparity
                    # flow here is actually disparity
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    # NOTE: reverse disp, disparity is positive
                    displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                    feature1 = flow_warp(feature1, displace)  # [B, C, H, W]
                elif task == 'flow':
                    feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                else:
                    raise NotImplementedError

            attn_splits = self.c.attn_splits_list[scale_idx]
            if task != 'depth':
                corr_radius = self.c.corr_radius_list[scale_idx]
            prop_radius = self.c.prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            # console.log(f"{scale_idx}: feature0 shape {feature0.shape}, -- attn_splits {attn_splits}, {corr_radius}, {prop_radius}")

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=self.c.attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

            # correlation and softmax
            if task == 'depth':
                # first generate depth candidates
                b, _, h, w = feature0.size()
                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(feature0)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h,
                                                                                               w)  # [B, D, H, W]

                flow_pred = correlation_softmax_depth(feature0, feature1,
                                                      intrinsics_curr,
                                                      pose,
                                                      depth_candidates=depth_candidates,
                                                      depth_from_argmax=depth_from_argmax,
                                                      pred_bidir_depth=pred_bidir_depth,
                                                      )[0]

            else:
                if corr_radius == -1:  # global matching
                    if task == 'flow':
                        flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
                    elif task == 'stereo':
                        flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
                    else:
                        raise NotImplementedError
                else:  # local matching
                    if task == 'flow':
                        flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
                    elif task == 'stereo':
                        flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]
                    else:
                        raise NotImplementedError

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            if task == 'stereo':
                flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if (pred_bidir_flow or pred_bidir_depth) and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation

            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution

                    if task == 'stereo':
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        flow_up_pad = self.upsample_flow(flow_pad, feature0)
                        flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
                    elif task == 'depth':
                        depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                          is_depth=True).clamp(min=min_depth, max=max_depth)
                        flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]
                    else:
                        flow_up = self.upsample_flow(flow, feature0)

                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert self.c.num_reg_refine > 0
                    for refine_iter_idx in range(self.c.num_reg_refine):
                        flow = flow.detach()

                        if task == 'stereo':
                            zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                        elif task == 'depth':
                            if pred_bidir_depth and refine_iter_idx == 0:
                                intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                                pose = torch.cat((pose, torch.inverse(pose)), dim=0)

                                feature0_ori, feature1_ori = torch.cat((feature0_ori, feature1_ori),
                                                                       dim=0), torch.cat((feature1_ori,
                                                                                          feature0_ori), dim=0)

                            flow_from_depth = compute_flow_with_depth_pose(1. / flow.squeeze(1),
                                                                           intrinsics_curr,
                                                                           extrinsics_rel=pose,
                                                                           )

                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow_from_depth,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        else:
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                                  )

                        if task == 'depth':
                            flow = (flow - residual_flow).clamp(min=min_depth, max=max_depth)
                        else:
                            flow = flow + residual_flow

                        if task == 'stereo':
                            flow = flow.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == self.c.num_reg_refine - 1:
                            if task == 'depth':
                                if refine_iter_idx < self.c.num_reg_refine - 1:
                                    # bilinear upsampling
                                    flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                                 upsample_factor=upsample_factor,
                                                                 is_depth=True)
                                else:
                                    # last one convex upsampling
                                    # NOTE: clamp depth due to the zero padding in the unfold in the convex upsampling
                                    # pad depth to 2 channels as flow
                                    depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                                    depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                                      is_depth=True).clamp(min=min_depth,
                                                                                           max=max_depth)
                                    flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]

                            else:
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')

                            flow_preds.append(flow_up)

        if task == 'stereo':
            for i in range(len(flow_preds)):
                flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]

        # convert inverse depth to depth
        if task == 'depth':
            for i in range(len(flow_preds)):
                flow_preds[i] = 1. / flow_preds[i].squeeze(1)  # [B, H, W]

        results_dict.update({'flow_preds': flow_preds})

        return results_dict

    def forward_correspondence(self, img0, img1):
        task = self.c.task
        results_dict = {}
        flow_preds = []

        # Step: 1) list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
            feature0_ori, feature1_ori = feature0, feature1
            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = self.c.attn_splits_list[scale_idx]
            corr_radius = self.c.corr_radius_list[scale_idx]
            prop_radius = self.c.prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            # console.log(f"{scale_idx}: feature0 shape {feature0.shape}, -- attn_splits {attn_splits}, {corr_radius}, {prop_radius}")

            # Transformer
            feature0, feature1 = self.transformer(
                feature0, feature1, attn_type=self.c.attn_type, attn_num_splits=attn_splits
            )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution
                    flow_up = self.upsample_flow(flow, feature0)
                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert self.c.num_reg_refine > 0
                    for refine_iter_idx in range(self.c.num_reg_refine):
                        flow = flow.detach()
                        correlation = local_correlation_with_flow(
                            feature0_ori,
                            feature1_ori,
                            flow=flow,
                            local_radius=4,
                        )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                                  )

                        flow = flow + residual_flow

                        if self.training or refine_iter_idx == self.c.num_reg_refine - 1:
                            flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                              is_depth=task == 'depth')
                            flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        return results_dict

    def forward_stereo(self, img0, img1):
        """
        all images should be normalized in the dataset
        """
        task = self.c.task
        results_dict = {}
        flow_preds = []

        # Step: 1) list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
            feature0_ori, feature1_ori = feature0, feature1
            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()

                # construct flow vector for disparity
                # flow here is actually disparity
                zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                # NOTE: reverse disp, disparity is positive
                displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                feature1 = flow_warp(feature1, displace)  # [B, C, H, W]

            attn_splits = self.c.attn_splits_list[scale_idx]
            corr_radius = self.c.corr_radius_list[scale_idx]
            prop_radius = self.c.prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(
                feature0, feature1, attn_type=self.c.attn_type, attn_num_splits=attn_splits
            )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred
            flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            flow = self.feature_flow_attn(
                feature0, flow.detach(), local_window_attn=prop_radius > 0, local_window_radius=prop_radius
            )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution
                    flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                    flow_up_pad = self.upsample_flow(flow_pad, feature0)
                    flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert self.c.num_reg_refine > 0
                    for refine_iter_idx in range(self.c.num_reg_refine):
                        flow = flow.detach()

                        zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                        # NOTE: reverse disp, disparity is positive
                        displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                        correlation = local_correlation_with_flow(
                            feature0_ori, feature1_ori, flow=displace, local_radius=4,
                        )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)
                        net, inp = torch.chunk(proj, chunks=2, dim=1)
                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone())
                        flow = flow + residual_flow
                        flow = flow.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == self.c.num_reg_refine - 1:
                            flow_up = upsample_flow_with_mask(
                                flow, up_mask, upsample_factor=self.upsample_factor, is_depth=task == 'depth'
                            )
                            flow_preds.append(flow_up)

        for i in range(len(flow_preds)):
            flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]

        results_dict.update({'flow_preds': flow_preds})

        return results_dict

    def get_image_info(self, image: Union[str, Path, torch.Tensor]):
        if not isinstance(image, torch.Tensor):
            image = load_rgb(image)
        if image.size(-2) > image.size(-1):
            image = torch.transpose(image, -2, -1)
            self.transpose_img = True

        self.raw_size = image.shape[-2:]
        self.inference_size = [
            int(ceil(image.size(-2) / self.c.padding_factor)) * self.c.padding_factor,
            int(ceil(image.size(-1) / self.c.padding_factor)) * self.c.padding_factor
        ]
        if self.inference_size[0] != self.raw_size[0] or self.inference_size[1] != self.raw_size[1]:
            self.resize_img = True
        console.print(f"transpose? {self.transpose_img}, resize? {self.resize_img}, to {self.inference_size}")

    def forward_flow(self, image1: torch.Tensor, image2: torch.Tensor):
        """
        assume the images to be float-type (already divided by 255) and normalized
        Args:
            image1: (b, c=3, h, w)
            image2: (b, c=3, h, w)
        """
        if self.transpose_img:
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
        if self.resize_img:
            image1 = F.interpolate(image1, size=self.inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size=self.inference_size, mode='bilinear', align_corners=True)

        # console.log(f"{image1.shape}, {image2.shape}")
        return self.forward_correspondence(image1, image2)

    def forward_disparity(self, left: torch.Tensor, right: torch.Tensor):
        fixed_inference_size = self.inference_size

        nearest_size = [int(np.ceil(left.size(-2) / self.c.padding_factor)) * self.c.padding_factor,
                        int(np.ceil(left.size(-1) / self.c.padding_factor)) * self.c.padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        ori_size = left.shape[-2:]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            left = F.interpolate(left, size=inference_size, mode='bilinear', align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear', align_corners=True)

        with torch.no_grad():
            pred_disp = self.forward_stereo(left, right)['flow_preds'][-1]  # [1, H, W]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)  # [1, H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        disp = pred_disp[0]
        return disp

    def get_depth(self, image1: torch.Tensor, image2: torch.Tensor, base_line_mm: float, focal_length: float):
        disparity = self.forward_disparity(image1, image2)
        depth_map = torch.divide(torch.tensor([base_line_mm * focal_length]).to(disparity), disparity + 1e-10)
        depth_map[depth_map > self.c.max_depth_in_mm] = self.c.max_depth_in_mm
        return depth_map

    def get_prob(self, image1: torch.Tensor, image2: torch.Tensor):
        if self.transpose_img:
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
        if self.resize_img:
            image1 = F.interpolate(image1, size=self.inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size=self.inference_size, mode='bilinear', align_corners=True)

        console.log(f"{image1.shape}, {image2.shape}")

        img0, img1 = image1, image2
        task = self.c.task
        results_dict = {}
        flow_preds = []

        # Step: 1) list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None
        probs = []
        features = []

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
            feature0_ori, feature1_ori = feature0, feature1
            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            # if scale_idx > 0:
            #     flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
            #
            # if flow is not None:
            #     flow = flow.detach()
            #     feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = self.c.attn_splits_list[scale_idx]
            corr_radius = self.c.corr_radius_list[scale_idx]
            prop_radius = self.c.prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            console.log(
                f"{scale_idx}: feature0 shape {feature0.shape}, -- attn_splits {attn_splits}, {corr_radius}, {prop_radius}")

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=self.c.attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

            feature0 = F.interpolate(feature0, size=self.raw_size, mode='bilinear', align_corners=True)
            feature1 = F.interpolate(feature1, size=self.raw_size, mode='bilinear', align_corners=True)
            features.append([feature0.detach().cpu(), feature1.detach().cpu()])
            # prob = global_prob(feature0.detach().cpu(), feature1.detach().cpu())
            # probs.append(prob)
            continue

            # correlation and softmax
            feature_shapes.append(feature0.shape)
            if corr_radius == -1:  # global matching
                flow_pred, prob = global_correlation_softmax(feature0, feature1)
                probs.append(prob.detach().cpu())
            else:  # local matching
                flow_pred, prob = local_correlation_softmax(feature0, feature1, corr_radius)

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution
                    flow_up = self.upsample_flow(flow, feature0)
                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert self.c.num_reg_refine > 0
                    for refine_iter_idx in range(self.c.num_reg_refine):
                        flow = flow.detach()
                        correlation = local_correlation_with_flow(
                            feature0_ori,
                            feature1_ori,
                            flow=flow,
                            local_radius=4,
                        )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                                  )

                        flow = flow + residual_flow

                        if self.training or refine_iter_idx == self.c.num_reg_refine - 1:
                            flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                              is_depth=task == 'depth')
                            flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds, "probs": probs, "features": features})

        return results_dict
