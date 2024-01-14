import cv2
import click
import numpy as np
import torch
from pathlib import Path
from typing import Union
from rich.table import Table
from marshmallow_dataclass import dataclass

import robot_utils.viz.latex_colors_rgba as lc
from robot_utils.py.utils import load_dataclass
from robot_utils.cv.opencv import draw_reticle, get_gaussian_kernel_heatmap
from robot_utils.cv.opencv import overlay_masks_on_image
from robot_utils.cv.correspondence.finder_torch import find_best_match_for_descriptor, find_best_match_for_descriptor_cosine

from robot_utils import console
from robot_utils.py.filesystem import copy2, validate_path
from robot_utils.torch.torch_utils import init_torch

from robot_vision.utils.utils import get_dcn_path
from robot_vision.dcn.wrapper import DCNWrapper
from robot_vision.dcn.dataset import PDCDataset
from robot_vision.grounded_sam.wrapper import MaskWrapper, MaskWrapperConfig


@dataclass
class VizConfig:
    image_blend_weight: float = 0.3
    heatmap_kernel_variance: float = 0.25
    heatmap_vis_upper_bound: float = 0.75  # Not used anymore

    mask_confident_lower_th: float = 0.1

    # The following for demo
    image_ext: str = ".jpg"
    depth_ext: str = ".png"
    n_images_per_demo: int = 20
    n_points_per_obj: int = 300
    ref_image_index: int = -1


class VizCorrespondence:
    def __init__(
            self,
            path_dcn: Union[str, Path],
            path_image_1: Union[str, Path] = None,
            path_image_2: Union[str, Path] = None,
            mode: str = "eval"
    ):
        self.root_path_dcn, _ = validate_path(path_dcn)
        self.root_path_image_1, self._enable_image_1 = validate_path(path_image_1)
        self.root_path_image_2, self._enable_image_2 = validate_path(path_image_2)
        self.device = init_torch(seed=2023, use_gpu=True)

        heatmap_config_file = self.root_path_dcn / "heatmap.yaml"
        if not heatmap_config_file.exists():
            src = get_dcn_path() / "config/heatmap.yaml"
            copy2(src, heatmap_config_file)

        console.log(f"use config: {heatmap_config_file}")
        self.c = load_dataclass(VizConfig, heatmap_config_file)

        self._load_dcn_model(mode)
        self._load_mask_model()

        self._reticle_color = (0, 255, 0)
        self._paused = False

        self.u = 0
        self.v = 0

        from robot_utils.cv.io.io_cv import load_rgb
        if self._enable_image_1:
            self.preload_img1 = self._dataset.get_rgb_from_path(self.root_path_image_1)
        if self._enable_image_2:
            self.preload_img2 = self._dataset.get_rgb_from_path(self.root_path_image_2)

        self._instruction()

    def _instruction(self):
        table = Table(title="Instructions")
        table.add_column("Key", justify="center", style="cyan")
        table.add_column("Functionality", justify="left", style="green")
        table.add_row("n", "sample a new pair of images")
        table.add_row("s", "switch the source and the target image")
        console.print(table)

    def _load_mask_model(self):
        mask_obj_list = [self.obj]
        cfg = MaskWrapperConfig()
        cfg.obj_prompt_cfg = mask_obj_list
        self.mask = MaskWrapper(cfg)
        self._masks = {}

    def _load_dcn_model(self, mode: str):
        self._dataset = PDCDataset(self.root_path_dcn / "data_config.yaml", mode=mode)
        self.obj = self._dataset.c.obj
        self.intrinsics = torch.from_numpy(self._dataset.intrinsics).float()

        self.dcn = DCNWrapper({self.obj: self.obj})

    def _fresh_window(self):
        self._load_img_pair()
        self._mask = self._compute_masks(self.img2)
        self._compute_descriptors()
        self.find_best_match(None, 0, 0, None, None)

    def _load_img_pair(self):
        """
        Gets a new pair of images
        """
        scene_name_1, scene_name_2 = self._dataset.get_random_scene(2)
        self.img1, self.img2 = None, None

        if self._enable_image_1:
            self.img1 = self.preload_img1
        if self._enable_image_2:
            self.img2 = self.preload_img2
        else:
            pass

        if self.img1 is None:
            image_idx_1 = self._dataset.get_random_image_index(scene_name_1)
            self.img1 = self._dataset.get_rgb(scene_name_1, image_idx_1)
        if self.img2 is None:
            image_idx_2 = self._dataset.get_random_image_index(scene_name_2)
            self.img2 = self._dataset.get_rgb(scene_name_2, image_idx_2)

        cv2.imshow('source', self.img1[..., ::-1])
        cv2.imshow('target', self.img2)

    def _compute_masks(self, image: np.ndarray):
        mask_obj_list = [self.obj]
        pred = self.mask.get_masks(image, 0, mask_obj_list)
        mask = pred["masks_np"].get(self.obj, None)
        if mask is None:
            return None
        return mask[0]

    def _compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        """
        self._res_a = self.dcn.compute_descriptor_images([self.img1])[self.obj]
        self._res_b = self.dcn.compute_descriptor_images([self.img2])[self.obj]

    # def find_best_match(self, event, u, v, flags, param):
    #     obj = "plate"
    #     path = Path(f"/home/gao/dataset/kvil/demo/demo_stack_spoon_bigger_scene/canonical/dcn/{obj}")
    #     uv_can_yaml = path / "uv.yaml"
    #     uv_can = np.array(load_dict_from_yaml(uv_can_yaml), dtype=int)
    #
    #     descriptors = self._res_a[uv_can[:, 1], uv_can[:, 0]]
    #     best_match_uv, best_match_diff, norm_diffs = find_best_match_for_descriptor(
    #         descriptors.to(self.dcn.device), self._res_b.to(self.dcn.device),
    #         mask=torch.tensor(self._mask_expand, dtype=torch.bool, device=self.dcn.device)
    #     )
    #     img = viz_points_with_color_map(self.img2, best_match_uv.detach().cpu().numpy(), copy_image=True)
    #     # if self.dcn.dcn
    #     cv2.imshow("target", img)

    def find_best_match(self, event, u, v, flags, param):
        if self._paused:
            return
        self.u = u
        self.v = v
        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, u, v, self._reticle_color)
        cv2.imshow("source", img_1_with_reticle[..., ::-1])

        img_2_with_reticle = np.copy(self.img2)
        img_2_with_reticle = overlay_masks_on_image(
            img_2_with_reticle, [self._mask], colors=[(np.array(lc.orange_peel[:3]) * 255).astype(int)], rgb_weights=0.5
        )

        res_a = self._res_a
        res_b = self._res_b
        # best_match_uv = find_best_match_for_descriptor_cosine(
        #     res_a[v, u], res_b, mask=torch.tensor(self._mask, dtype=torch.bool), is_softmax=False
        # )
        best_match_uv, best_match_diff, norm_diffs = find_best_match_for_descriptor(
            res_a[v, u], res_b, mask=torch.tensor(self._mask, dtype=torch.bool)
        )
        heatmap_color = get_gaussian_kernel_heatmap(norm_diffs.numpy(), self.c.heatmap_kernel_variance)
        best_match_uv = best_match_uv.numpy().astype(int)

        reticle_color = [0, 0, 255]
        draw_reticle(img_2_with_reticle, best_match_uv[0], best_match_uv[1], reticle_color)
        cv2.imshow("target", img_2_with_reticle[..., ::-1])

        draw_reticle(heatmap_color, best_match_uv[0], best_match_uv[1], reticle_color)
        alpha = self.c.image_blend_weight
        beta = 1 - alpha
        blended = cv2.addWeighted(self.img2, alpha, heatmap_color, beta, 0)

        cv2.imshow("heatmap", blended)
        cv2.moveWindow('heatmap', self._dataset.image_wh[0], self._dataset.image_wh[1] + 100)

    def move_window(self):
        cv2.namedWindow('target')
        cv2.moveWindow('target', self._dataset.image_wh[0], 20)
        cv2.setMouseCallback('source', self.find_best_match)
        cv2.moveWindow('source', 0, 20)

    def _switch_img(self):
        self.img1, self.img2 = self.img2, self.img1
        self._mask = self._compute_masks(self.img2)
        self._compute_descriptors()
        self.find_best_match(None, 0, 0, None, None)

    def run(self):
        self._fresh_window()
        self.move_window()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                self._fresh_window()

            elif k == ord('s'):
                self._switch_img()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path_dcn",     "-p",       type=str,   help="the absolute path to the training root folder of dcn")
@click.option("--path_image_1", "-pi1",     type=str,   help="the absolute path to the standalone test image")
@click.option("--path_image_2", "-pi2",     type=str,   help="the absolute path to the standalone test image")
@click.option("--mode",         "-m",       type=str,   default="eval", help="mode of the dcn dataset")
def main(path_dcn, path_image_1, path_image_2, mode):
    try:
        heatmap_vis = VizCorrespondence(
            path_dcn, path_image_1, path_image_2, mode
        )
        heatmap_vis.run()
    finally:
        console.rule("done")


if __name__ == "__main__":
    main()
