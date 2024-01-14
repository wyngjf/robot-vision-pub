import sys
import os
import json
import cv2
import click
import numpy as np
from typing import Union, Tuple
from pathlib import Path
from rich.progress import track

from robot_utils import console
from robot_utils.py.utils import load_dict_from_yaml, save_to_yaml
from robot_utils.py.filesystem import create_path
from robot_utils.cv.geom.projection import pinhole_projection_image_to_world
from robot_vision.dataset.dex_nerf.utils import get_intrinsics_from_transform
from robot_utils.cv.io.io_cv import load_depth
from scipy.spatial.transform import Rotation as R

NGP_PATH = os.environ.get("NGP_PATH")
if not NGP_PATH:
    console.rule("[bold red]missing DEX_NGP_PATH env variable")
    exit(1)
sys.path.append(str(Path(NGP_PATH, "scripts")))
import pyngp as ngp
from common import *


def generate_masks(
        path: Union[str, Path],
        namespace: str,
        rand_rotation: bool = False,
        depth_path_for_mask: Path = None
):
    path = Path(path)
    console.rule("[blue] Generate Masks")
    console.print(f"under scene: {path}")
    if depth_path_for_mask is not None and depth_path_for_mask.is_dir():
        path_depth = depth_path_for_mask
    else:
        if rand_rotation:
            path_depth = path.parent / f"{path.stem}_rand_rot" / namespace / "depth"
            # path_mask = create_path(path_depth.parent / "mask", remove_existing=True)
        else:
            path_depth = path / namespace / "depth"
    path_mask = create_path(path / namespace / "mask", remove_existing=False)
    pose_file = path / namespace / "pose_data.yaml"
    aabb_file = path / namespace / "crop_box.yaml"

    transforms_file = path / "transforms.json"
    with open(transforms_file) as f:
        transforms = json.load(f)
        offset = np.array(transforms.get("offset", [0.5, 0.5, 0.5]))
        scale = transforms.get("scale", 0.33)
        w_max, h_max = int(transforms["w"]), int(transforms["h"])

    ws = np.arange(w_max)
    hs = np.arange(h_max)
    hw = np.stack(np.meshgrid(hs, ws, indexing="ij"), axis=-1).reshape(-1, 2)
    wh = hw[:, [1, 0]].astype(int)

    intrinsics = get_intrinsics_from_transform(transforms)

    poses = load_dict_from_yaml(pose_file)["poses"]
    n_images = len(poses)

    aabb_dict = load_dict_from_yaml(aabb_file)
    aabb = (np.array(aabb_dict["crop_box_corners"]) - offset) / scale
    aabb_rotation = np.array(aabb_dict["aabb_rotation"])
    aabb = np.einsum("ij,bj->bi", aabb_rotation, aabb)
    lower = aabb[0]
    upper = aabb[-1]

    aabb_file = path / namespace / "aabb.yaml"
    save_to_yaml(dict(lower=lower.tolist(), upper=upper.tolist(), rotation=aabb_rotation.tolist()), aabb_file)

    for i in track(range(n_images), description="[blue]Generating masks:"):
        z = load_depth(path_depth / f"{i:04d}.png")
        pose = np.array(poses[i])
        pcl = pinhole_projection_image_to_world(wh, z, pose, intrinsics).squeeze()
        pcl = np.einsum("ij,bj->bi", aabb_rotation, pcl)
        mask = np.all(np.logical_and(lower <= pcl, upper >= pcl), axis=1).astype(int).reshape((h_max, w_max))
        cv2.imwrite(str(path_mask/f"{i:04d}.png"), mask)
        cv2.imwrite(str(path_mask/f"visible_{i:04d}.png"), mask * 255)


def render_mask_from_ngp(
        scene_dir: Path,
        snapshot_file: Path = "base_mask.msgpack",
        sigma_threshold=15,
        namespace: str = "base",
        training_view: bool = True,
        video_n_seconds: int = 1,
        video_fps: int = 30,
        enable_rand_rot: bool = False,
        rand_rot_range_degree: Tuple[float, float] = (-120, 120),
        rand_rot_matrices: np.ndarray = None,
) -> Path:
    mode = "depth"
    snapshot_file = Path(snapshot_file)
    if not snapshot_file.is_file():
        raise FileExistsError(f"snapshot {snapshot_file} does not exist")

    console.rule(f"[bold blue]rendering NeRF in mode {mode}")

    transforms_file = scene_dir / "transforms.json"
    console.print(f"loading {transforms_file}")
    with open(transforms_file, 'r') as tf:
        meta = json.load(tf)

    width = int(meta['w'])
    height = int(meta['h'])

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.shall_train = False
    testbed.nerf.sharpen = float(0.)
    testbed.load_training_data(str(scene_dir))
    testbed.dex_nerf = False
    testbed.render_mode = ngp.Depth
    testbed.dex_nerf = True
    testbed.sigma_thrsh = sigma_threshold
    testbed.nerf.rendering_min_transmittance = 1e-4
    img_ext = ".png"  # here we use png to be able to directly same 16bit depth map

    console.print(f"Loading snapshot {snapshot_file}")
    testbed.load_snapshot(str(snapshot_file))
    testbed.nerf.render_with_camera_distortion = True
    testbed.snap_to_pixel_centers = True
    spp = 1

    testbed.fov_axis = 0
    testbed.fov = meta["fovx"]  # camera_angle_x * 180 / np.pi
    testbed.fov_axis = 1
    testbed.fov = meta["fovy"]  # camera_angle_y * 180 / np.pi

    scale = testbed.nerf.training.dataset.scale
    offset = testbed.nerf.training.dataset.offset

    if training_view:
        n_frames = testbed.nerf.training.dataset.n_images
        testbed.camera_smoothing = False
    else:
        testbed.load_camera_path(str(scene_dir / namespace / "base_cam.json"))
        n_frames = video_n_seconds * video_fps

    if n_frames < 1:
        console.print(f"[bold red]Error: number of frames to be rendered is {n_frames}, please double check.")
        exit(1)

    if enable_rand_rot:
        render_path = create_path(scene_dir.parent / f"{scene_dir.stem}_rand_rot") / namespace / f"temp/depth_for_{mode}"
        if rand_rot_matrices is None:
            rand_rot_matrices = R.from_euler(
                "z", np.random.uniform(rand_rot_range_degree[0], rand_rot_range_degree[1], n_frames), degrees=True
            ).as_matrix()
    else:
        render_path = scene_dir / namespace / f"temp/depth_for_{mode}"

    render_poses = []
    identity_matrix = np.identity(4, dtype=float)

    console.print(f"render to path {render_path}")
    render_path = create_path(render_path, remove_existing=True)

    for i in track(range(n_frames), description="[blue]Rendering", console=console):
        if training_view:
            testbed.set_camera_to_training_view(i)
            raw = testbed.render(width, height, spp, True)  # if depth, values (float, in m)
            identity_matrix[:3] = testbed.camera_matrix
        else:
            key_frame = testbed.get_camera_from_time(float(i) / n_frames)

            qvec = key_frame.R
            tvec = key_frame.T
            mat = R.from_quat(qvec).as_matrix()
            if enable_rand_rot:
                mat = mat.dot(rand_rot_matrices[i])
            identity_matrix[:3, :3] = mat
            identity_matrix[:3, -1] = np.array(tvec)

            testbed.set_ngp_camera_matrix(identity_matrix[:3])
            identity_matrix[:3, -1] = (identity_matrix[:3, -1] - offset) / scale
            raw = testbed.render(width, height, spp, True)

        render_poses.append(identity_matrix.tolist())

        raw = raw[..., 0]
        depth_int = 1000 * raw  # transform depth to mm
        depth_int = depth_int.astype(np.uint16)
        cv2.imwrite(str(render_path / f"{i:>04d}{img_ext}"), depth_int)

    return render_path


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",         "-p",   type=str,   help="the path to the working directory of NGP")
@click.option("--namespace",    "-n",   type=str,   default="base", help="the namespace in which the rendered image data are stored")
def main(path, namespace):
    generate_masks(path, namespace)


if __name__ == "__main__":
    main()
