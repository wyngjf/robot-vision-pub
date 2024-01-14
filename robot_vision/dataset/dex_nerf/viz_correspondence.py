import cv2
import click
import logging
import json
import numpy as np
from pathlib import Path
from rich.table import Table
from robot_utils import console

from robot_utils.cv.io.io_cv import load_rgb, load_depth, load_mask
from robot_utils.py.utils import load_dict_from_yaml
from robot_utils.cv.opencv import draw_reticle
from robot_utils.cv.correspondence.finder_torch import pixel_correspondences, create_non_correspondences

from robot_vision.dataset.dex_nerf.utils import get_intrinsics_from_transform


class VizCorrespondence(object):
    def __init__(self, path: Path, namespace: str, enable_non_correspondence: bool = False):
        self.working_dir = path / namespace
        self.enable_non_correspondence = enable_non_correspondence

        self.path_shade = path / namespace / "shade"
        self.path_depth = path / namespace / "depth"
        self.path_mask = path / namespace / "mask"
        self.pose_file = path / namespace / "pose_data.yaml"

        transforms_file = path / "transforms.json"
        with open(transforms_file) as f:
            transforms_dict = json.load(f)

        self.poses = load_dict_from_yaml(self.pose_file)["poses"]
        self.n_images = len(self.poses)

        self.width = int(transforms_dict.get("w", 1280))
        self.height = int(transforms_dict.get("h", 720))
        scale = 0.6 if self.width > 1000 else 1.0
        self.non_match_th = int(50 * scale)
        self.image_wh = (int(scale * self.width), int(scale * self.height))

        console.print(f"scale: {scale}")
        self.intrinsics = get_intrinsics_from_transform(transforms_dict, scale=scale)

        self._reticle_color = (0, 255, 0)
        self._paused = False
        self._instruction()

    def _instruction(self):
        table = Table(title="Instructions")
        table.add_column("Key", justify="center", style="cyan")
        table.add_column("Functionality", justify="left", style="green")
        table.add_row("n", "sample a new pair of images")
        table.add_row("s", "switch the source and the target image")
        table.add_row("p", "pause/un-pause")
        console.print(table)

    def fresh_window(self):
        self._load_img_pair()
        self.find_best_match(None, 0, 0, None, None)

    def _load_img_pair(self):
        """
        Gets a new pair of images
        """
        idx_a = np.random.randint(self.n_images)
        idx_b = np.random.randint(self.n_images)
        console.print(f"indices of images: {idx_a}, {idx_b}")

        self.rgb_1 = cv2.resize(load_rgb(self.path_shade / f"{idx_a:04d}.jpg"), self.image_wh)
        self.rgb_2 = cv2.resize(load_rgb(self.path_shade / f"{idx_b:04d}.jpg"), self.image_wh)

        self.z_a = cv2.resize(load_depth(self.path_depth / f"{idx_a:04d}.png"), self.image_wh)
        self.z_b = cv2.resize(load_depth(self.path_depth / f"{idx_b:04d}.png"), self.image_wh)
        if self.z_a.ndim == 2:
            self.z_a = self.z_a[..., np.newaxis]
            self.z_b = self.z_b[..., np.newaxis]

        self.mask_a = cv2.resize(load_mask(self.path_mask / f"{idx_a:04d}.png"), self.image_wh)
        self.mask_b = cv2.resize(load_mask(self.path_mask / f"{idx_b:04d}.png"), self.image_wh)

        self.cam_pose_a = np.array(self.poses[idx_a]).reshape(4, 4)
        self.cam_pose_b = np.array(self.poses[idx_b]).reshape(4, 4)

    def find_best_match(self, event, u, v, flags, param):
        """
        For each network, find the best match in the target image to point highlighted with reticle in the source image. Displays the result
        """
        if self._paused:
            return

        uvs_a = np.array([u, v], dtype=int)[np.newaxis, ...]
        uvs_a, uvs_b = pixel_correspondences(
            self.z_a, self.cam_pose_a, self.z_b, self.cam_pose_b, self.intrinsics, uvs_a, 20, 'cpu', None,
            img_b_mask=self.mask_b,
            ignore_fov_occlusion=False
        )
        img_1_with_reticle = np.copy(self.rgb_1)
        img_2_with_reticle = np.copy(self.rgb_2)

        if uvs_b is None or uvs_b.nelement() == 0:
            cv2.imshow("source", img_1_with_reticle)
            cv2.imshow("target", img_2_with_reticle)
            return
        # uvs_b = pinhole_projection_image_to_image(uvs_a, self.z_a, self.cam_pose_a, self.cam_pose_b, self.intrinsics)
        uvs_b = uvs_b.squeeze()
        uvs_a, uvs_b = uvs_a.numpy().astype(int), uvs_b.numpy().astype(int)

        draw_reticle(img_1_with_reticle, uvs_a[0, 0], uvs_a[0, 1], self._reticle_color)
        cv2.imshow("source", img_1_with_reticle)

        reticle_color = [255, 255, 0]
        draw_reticle(img_2_with_reticle, uvs_b[0], uvs_b[1], self._reticle_color)
        if self.enable_non_correspondence:
            uv_b_non_match = create_non_correspondences(uvs_b, self.z_b.shape, 100, self.mask_b, self.non_match_th, True)
            uv_b_non_match = uv_b_non_match.numpy().astype(int)
            if uv_b_non_match is not None:
                uv_b_non_match = uv_b_non_match[0]
                for i in range(uv_b_non_match.shape[0]):
                    draw_reticle(img_2_with_reticle, uv_b_non_match[i, 0], uv_b_non_match[i, 1], reticle_color)
        cv2.imshow("target", img_2_with_reticle)

    def move_window(self):
        cv2.namedWindow('target')
        cv2.moveWindow('target', self.image_wh[0], 100)
        cv2.setMouseCallback('source', self.find_best_match)
        cv2.moveWindow('source', 0, 100)

    def _switch_img(self):
        self.rgb_1, self.rgb_2 = self.rgb_2, self.rgb_1
        self.z_a, self.z_b = self.z_b, self.z_a
        self.mask_a, self.mask_b = self.mask_b, self.mask_a
        self.cam_pose_a, self.cam_pose_b = self.cam_pose_b, self.cam_pose_a
        self.find_best_match(None, 0, 0, None, None)

    def run(self):
        self.fresh_window()
        self.move_window()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                self.fresh_window()
            elif k == ord('s'):
                self._switch_img()
            elif k == ord('p'):
                if self._paused:
                    logging.info("un pausing")
                    self._paused = False
                else:
                    logging.info("pausing")
                    self._paused = True


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",      "-p",  type=str,   help="the path to the working directory of NGP")
@click.option("--namespace", "-n",  type=str,   default="base", help="the namespace in which the rendered image data are stored")
@click.option("--enable_non_correspondence", "-nc",  is_flag=True, help="the namespace in which the rendered image data are stored")
def main(path, namespace, enable_non_correspondence):
    path = Path(path)
    console.rule("[blue] Examine Correspondence")
    console.print(f"under scene: {path}")

    try:
        heatmap_vis = VizCorrespondence(path, namespace, enable_non_correspondence)
        heatmap_vis.run()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
