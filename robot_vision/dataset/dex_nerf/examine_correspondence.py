import cv2
import json
import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from robot_utils import console
from robot_utils.py.utils import load_dict_from_yaml
from robot_utils.cv.io.io_cv import load_depth, load_mask, load_rgb
from robot_utils.cv.geom.projection import pinhole_projection_image_to_image
from robot_vision.dataset.dex_nerf.utils import get_intrinsics_from_transform


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",      "-p",  type=str,   help="the path to the working directory of NGP")
@click.option("--namespace", "-n",  type=str,   default="base", help="the namespace in which the rendered image data are stored")
def main(path, namespace):
    path = Path(path)
    console.rule("[blue] Examine Correspondence")
    console.print(f"under scene: {path}")
    path_shade = path / namespace / "shade"
    path_depth = path / namespace / "depth"
    path_mask = path / namespace / "mask"
    pose_file = path / namespace / "pose_data.yaml"

    transforms_file = path / "transforms.json"
    with open(transforms_file) as f:
        transforms = json.load(f)

    poses = load_dict_from_yaml(pose_file)["poses"]
    n_images = len(poses)
    # offset = np.array(transforms.get("offset", [0.5, 0.5, 0.5]))
    # scale = transforms.get("scale", 0.33)

    idx_a = np.random.randint(n_images)
    idx_b = np.random.randint(n_images)
    console.print(f"indices of images: {idx_a}, {idx_b}")

    rgb_a = load_rgb(path_shade / f"{idx_a:04d}.jpg")
    rgb_b = load_rgb(path_shade / f"{idx_b:04d}.jpg")

    z_a = load_depth(path_depth / f"{idx_a:04d}.png")
    z_b = load_depth(path_depth / f"{idx_b:04d}.png")
    if z_a.ndim == 2:
        z_a = z_a[..., np.newaxis]
        z_b = z_b[..., np.newaxis]

    mask_a = load_mask(path_mask / f"{idx_a:04d}.png")
    mask_b = load_mask(path_mask / f"{idx_b:04d}.png")

    # weight_a = np.ones_like(z_a, dtype=float)
    # weight_b = np.ones_like(z_b, dtype=float)

    cam_pose_a = np.array(poses[idx_a]).reshape(4, 4)
    cam_pose_b = np.array(poses[idx_b]).reshape(4, 4)

    intrinsics = get_intrinsics_from_transform(transforms)

    fig, ax = plt.subplots(3, 2, figsize=(18, 12))
    ax[0, 0].imshow(rgb_a)
    ax[0, 1].imshow(rgb_b)
    ax[1, 0].imshow(z_a)
    ax[1, 1].imshow(z_b)
    ax[2, 0].imshow(mask_a)
    ax[2, 1].imshow(mask_b)
    plot_canvas1 = ax[0, 0].figure.canvas
    plot_canvas2 = ax[0, 1].figure.canvas
    plt.draw()
    plt.pause(0.01)

    def draw_images(w_a, h_a):
        uvs_a = np.array([w_a, h_a])[np.newaxis, ...]
        uvs_b = pinhole_projection_image_to_image(uvs_a, z_a, cam_pose_a, cam_pose_b, intrinsics)

        color = [255, 255, 0]
        img_a_draw = rgb_a.copy()
        img_a_draw = cv2.circle(img_a_draw, tuple([w_a, h_a]), 10, color, -1)

        img_b_draw = rgb_b.copy()
        img_b_draw = cv2.circle(img_b_draw, tuple(uvs_b.astype(int)), 15, color, -1)
        # img_B_draw = cv2.circle(img_B_draw, (uvs_B[1], uvs_B[0]), 15, color, -1)
        # for idx_pt in range(uvs_B.shape[0]):
        #     for idx_depth in range(uvs_B.shape[1]):
        #         ic(tuple(uvs_B[idx_pt][idx_depth]), idx_depth)
        #         img_B_overlay = img_B_draw.copy()
        #         img_B_overlay = cv2.circle(img_B_overlay, tuple(uvs_B[idx_pt][idx_depth]), 15, color, -1)
        #         alpha = weight_a[h_a, w_a][idx_depth] * 10
        #         img_B_draw = cv2.addWeighted(img_B_overlay, alpha, img_B_draw, 1 - alpha, 0)
        ax[0, 0].imshow(img_a_draw)
        ax[0, 1].imshow(img_b_draw)
        plot_canvas1.draw_idle()
        plot_canvas2.draw_idle()

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        u_a, v_a = int(event.xdata), int(event.ydata)
        draw_images(u_a, v_a)

    cid = plot_canvas1.mpl_connect('motion_notify_event', onclick)
    plt.show()


if __name__ == "__main__":
    main()
