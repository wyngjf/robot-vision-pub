import sys
import os
import json
import cv2
import numpy as np
from typing import Literal, Tuple
from pathlib import Path
from rich.progress import track

from robot_utils import console
from robot_utils.py.filesystem import create_path
from robot_utils.py.utils import save_to_yaml
from robot_utils.cv.io.io import image_to_video
from scipy.spatial.transform import Rotation as R
NGP_PATH = os.environ.get("NGP_PATH")
if not NGP_PATH:
    console.rule("[bold red]missing DEX_NGP_PATH env variable")
    exit(1)
sys.path.append(str(Path(NGP_PATH, "scripts")))
import pyngp as ngp
from common import *


def generate_camera_path_training_view(scene_dir: Path, namespace: str = "base", stereo: bool = False):
    console.rule("[bold blue]Generating camera path from transforms.json")
    transforms_file = scene_dir / "transforms.json"
    console.print(f"loading {transforms_file}")
    with open(transforms_file, 'r') as tf:
        train_transforms = json.load(tf)

    frames = train_transforms["frames"]
    n_train_sample = len(frames)
    console.print(f"in total {n_train_sample} frames")
    scale = train_transforms["scale"]
    offset = train_transforms["offset"]

    fov = 0.5 * (train_transforms["fovx"] + train_transforms["fovy"])
    xforms = {}
    for i in range(n_train_sample):
        file_name = frames[i]['file_path'].split('/')[-1][:-4]
        if stereo:
            file_name = file_name[:6]
        # file = int(file_name)
        xform = frames[i]['transform_matrix']
        xforms[file_name] = np.array(xform)

    keys = sorted(xforms.keys())
    cam_traj = {"path": []}
    for i in keys:
        mat = xforms[i]
        mat[:3, 3] = mat[:3, 3] * scale + offset
        rm = R.from_matrix(mat[:3, :3])
        quat, trans = rm.as_quat(), mat[:, 3]

        cam_traj["path"].append({
            "R": list(quat), "T": list(trans),
            "aperture_size": 0.0,
            "fov": fov,
            "glow_mode": 0.0,
            "glow_y_cutoff": 0.0,
            "scale": 0.0,
            "slice": 0.0,
            "dof": 0.0
        })

    video_camera_file = create_path(scene_dir / namespace) / "base_cam.json"
    console.print(f"saving to {video_camera_file}")
    with open(video_camera_file, 'w', encoding='utf-8') as f:
        json.dump(cam_traj, f, ensure_ascii=False, indent=4)


def get_crop_box(
        scene_dir: Path,
        snapshot_file: Path = "base.msgpack",
        namespace: str = "base",
) -> Path:
    snapshot_file = Path(snapshot_file)
    if not snapshot_file.is_file():
        raise FileExistsError(f"snapshot {snapshot_file} does not exist")

    console.rule(f"[bold blue]reading crop box from {snapshot_file}")
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.shall_train = False
    testbed.load_snapshot(str(snapshot_file))

    crop_box = testbed.crop_box(False)
    crop_box_corners = testbed.crop_box_corners(False)

    # The crop box you get from ngp is in the same world coordinate system as the point cloud (depth maps).
    # in the case you have a oriented bounding box (OBB), you cannot directly compare the PCL with the OBB in
    # each dimension, you should first rotate both the PCL and OBB, so that the edges of the OBB align with
    # the world coordinate system. This is why we also need the `render_aabb_to_local` variable here, which
    # is the orientation of the OBB in world coordinate system.
    render_aabb_to_local = testbed.render_aabb_to_local
    # ic(crop_box, crop_box_corners, render_aabb_to_local)

    aabb = dict(
        crop_box=crop_box.tolist(),
        crop_box_corners=[c.tolist() for c in crop_box_corners],
        aabb_rotation=render_aabb_to_local.tolist(),
    )
    aabb_filename = create_path(scene_dir / namespace) / "crop_box.yaml"
    save_to_yaml(aabb, aabb_filename)
    console.print(f"[green]crop box saved to {aabb_filename}")
    return aabb_filename


def render_ngp(
        scene_dir: Path,
        snapshot_file: Path = "base.msgpack",
        sigma_threshold=15,
        namespace: str = "base",
        mode: Literal["shade", "depth", "position"] = "shade",
        training_view: bool = True,
        video_n_seconds: int = 1,
        video_fps: int = 30,
        save_video: bool = True,
        save_pose: bool = True,
        enable_rand_rot: bool = False,
        rand_rot_range_degree: Tuple[float, float] = (-120, 120),
        rand_rot_matrices: np.ndarray = None,
):
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
    exposure = float(0.)
    img_ext = ".jpg"

    if mode == "shade":
        testbed.render_mode = ngp.Shade
        testbed.exposure = exposure

    elif mode == "depth":
        testbed.render_mode = ngp.Depth
        testbed.dex_nerf = True
        testbed.sigma_thrsh = sigma_threshold
        testbed.nerf.rendering_min_transmittance = 1e-4
        img_ext = ".png"  # here we use png to be able to directly same 16bit depth map

    elif mode == "position":
        testbed.render_mode = ngp.Positions
    else:
        raise ValueError(f"{mode} is not supported yet")

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
        render_path = create_path(scene_dir.parent / f"{scene_dir.stem}_rand_rot") / namespace / mode
        if rand_rot_matrices is None:
            rand_rot_matrices = R.from_euler(
                "z", np.random.uniform(rand_rot_range_degree[0], rand_rot_range_degree[1], n_frames), degrees=True
            ).as_matrix()
    else:
        render_path = scene_dir / namespace / mode

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

        if mode == "depth":
            raw = raw[..., 0]
            depth_int = 1000 * raw  # transform depth to mm
            depth_int = depth_int.astype(np.uint16)
            cv2.imwrite(str(render_path / f"{i:>04d}{img_ext}"), depth_int)
        else:
            write_image(str(render_path / f"{i:>04d}{img_ext}"), np.clip(raw * 2 ** exposure, 0.0, 1.0), quality=100)

    if save_video:
        video_file = render_path.parent / f"{mode}.mp4"
        vid_mode = "depth" if mode == "depth" else "rgb"
        image_to_video(render_path, video_file, fps=video_fps, codec="mp4v", mode=vid_mode)
        # img_file = render_path / f"%04d{img_ext}"
        # run(f"ffmpeg -y -framerate {video_fps} -i {img_file} -c:v libx264 -pix_fmt yuv420p {video_file}")

    if save_pose:
        save_to_yaml(dict(poses=render_poses), render_path.parent / "pose_data.yaml")

    ngp.free_temporary_memory()
    return rand_rot_matrices if enable_rand_rot else None

