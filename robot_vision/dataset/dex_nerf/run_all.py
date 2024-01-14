import click
import numpy as np
import cv2
from typing import List
from pathlib import Path
from robot_utils import console
from robot_utils.cv.io.io import extract_images_from_video
from robot_vision.dataset.dex_nerf.transforms import generate_transforms
from robot_utils.cv.colmap import run_colmap
from robot_utils.py.interact import ask_checkbox
from robot_utils.py.filesystem import get_ordered_subdirs, get_ordered_files
from robot_vision.dataset.dex_nerf.train_nerf import train_ngp
from robot_vision.dataset.dex_nerf.render_nerf import render_ngp, generate_camera_path_training_view, get_crop_box
from robot_vision.dataset.dex_nerf.generate_masks import generate_masks, render_mask_from_ngp
from robot_vision.dataset.dex_nerf.viz_nerf import viz_testbed
from robot_vision.dataset.dex_nerf.viz_correspondence import VizCorrespondence

import importlib
try:
    zed_found = importlib.util.find_spec("pyzed") is not None
    if zed_found:
        from robot_vision.dataset.stereo.zed.exporter import export
except:
    console.log("[bold red]ZED not working")


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",             "-p",   type=str,       help="the path to the object root")
@click.option("--namespace",        "-n",   type=str,       default="kvil", help="namespace to save rendered data")
@click.option("--epochs",           "-e",   type=int,       default=5000, help="iterations to train")
@click.option("--aabb",             "-ab",  type=int,       default=8, help="iterations to train")
@click.option("--n_frames",         "-f",   type=int,       default=100, help="iterations to train")
@click.option("--sigma_threshold",  "-s",   type=int,       default=15, help="iterations to train")
@click.option("--extract",          "-x",   is_flag=True,   help="to extract images from video?")
@click.option("--colmap",           "-c",   is_flag=True,   help="to run colmap?")
@click.option("--gen_transform",    "-g",   is_flag=True,   help="to generate transforms?")
@click.option("--train",            "-t",   is_flag=True,   help="to train?")
@click.option("--render",           "-r",   is_flag=True,   help="to render?")
@click.option("--viz",              "-v",   is_flag=True,   help="to only visualize?")
@click.option("--viz_mask",         "-vm",  is_flag=True,   help="to only visualize the cropped scene?")
@click.option("--viz_corresp",      "-vc",  is_flag=True,   help="to visualize the correspondence?")
@click.option("--viz_non_corresp",  "-vcn", is_flag=True,   help="to visualize the non-correspondence?")
@click.option("--render_mask",      "-m",   is_flag=True,   help="to render masks?")
@click.option("--train_view",               is_flag=True,   help="to render with cam?")
@click.option("--stereo",                   is_flag=True,   help="recorded with stereo camera?")
@click.option("--rand_rot",                 is_flag=True,   help="to render with random rotation?")
@click.option("--rand_range",               type=(float, float), default=(-120, 120),   help="the range of the random rotation as tuple of degrees")
@click.option("--fps",                      type=int,       default=30, help="fps of the rendered video")
@click.option("--video_len",                type=int,       default=1, help="time in seconds of the rendered video")
def main(path, namespace, epochs, aabb, n_frames, sigma_threshold, extract, colmap, gen_transform, train, render, viz,
         viz_mask, viz_corresp, viz_non_corresp, render_mask, train_view, fps, video_len, stereo, rand_rot, rand_range):
    path = Path(path)
    all_scenes = ask_checkbox("choose scenes to proceed", get_ordered_subdirs(path))  # type: List[Path]
    for p in all_scenes:
        if extract:
            if stereo:
                if not zed_found:
                    console.log(f"[bold red]You want to extract with stereo ZED camera, but sdk is not found.")
                    exit(1)
                svo = get_ordered_files(p, pattern=[".svo"])[0]
                export(filename=str(svo), mode=2, tn_frames=n_frames)
            else:
                video = get_ordered_files(p, pattern=[".mkv"])[0]
                extract_images_from_video(video, target_frames=n_frames)
        if colmap:
            run_colmap(p / "images")

        if gen_transform:
            generate_transforms(p, aabb)

        snapshot_file = p / "base.msgpack"
        if train:
            train_ngp(p, n_steps=epochs, snapshot_file=snapshot_file)

        if viz or viz_mask:
            viz_snapshot_file = p / "base_mask.msgpack" if viz_mask else snapshot_file
            if not viz_snapshot_file.is_file():
                console.rule(f"[bold red]skip {p}, snapshot doesn't exist")
            viz_testbed(p, snapshot_file=viz_snapshot_file)

        rand_rot_mat_filename = p / namespace / "rand_rot_mat.npy"
        if render:
            if not train_view:
                generate_camera_path_training_view(p, namespace, stereo)
            rand_rot_mat = render_ngp(
                p, snapshot_file, sigma_threshold, namespace,
                mode="shade", training_view=train_view, video_fps=fps, video_n_seconds=video_len,
                enable_rand_rot=rand_rot, rand_rot_range_degree=rand_range
            )
            if rand_rot_mat is not None and isinstance(rand_rot_mat, np.ndarray):
                np.savez(str(rand_rot_mat_filename), rand_rot_mat)

            render_ngp(
                p, snapshot_file, sigma_threshold, namespace,
                mode="depth", training_view=train_view, video_fps=fps, video_n_seconds=video_len,
                enable_rand_rot=rand_rot, rand_rot_range_degree=rand_range, rand_rot_matrices=rand_rot_mat
            )
            get_crop_box(p, snapshot_file, namespace)

        if render_mask:
            # first fetch aabb and then use depth maps to generate masks
            snapshot_file = p / "base_mask.msgpack"
            get_crop_box(p, snapshot_file, namespace)
            # if rand_rot_mat_filename.is_file():
            #     rand_rot_mat = np.load(str(rand_rot_mat_filename))
            # else:
            #     rand_rot_mat = None
            # depth_path = render_mask_from_ngp(
            #     p, snapshot_file, sigma_threshold, namespace,
            #     training_view=train_view, video_fps=fps, video_n_seconds=video_len,
            #     enable_rand_rot=rand_rot, rand_rot_range_degree=rand_range, rand_rot_matrices=rand_rot_mat
            # )
            # generate_masks(p, namespace, rand_rot, depth_path)
            generate_masks(p, namespace, rand_rot, None)

        if viz_corresp:
            try:
                heatmap_vis = VizCorrespondence(p, namespace, viz_non_corresp)
                heatmap_vis.run()
            finally:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
