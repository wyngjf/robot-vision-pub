import click
from pathlib import Path
from robot_utils.cv.io.io import extract_images_from_video
from robot_vision.dataset.dex_nerf.transforms import generate_transforms
from robot_utils.cv.colmap import run_colmap
from robot_vision.dataset.dex_nerf.train_nerf import train_ngp
from robot_vision.dataset.dex_nerf.render_nerf import render_ngp, generate_camera_path_training_view, get_crop_box
from robot_vision.dataset.dex_nerf.generate_masks import generate_masks


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",             "-p",   type=str,       help="the path to the data root")
@click.option("--video",            "-v",   type=str,       default="", help="absolute path to a video")
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
@click.option("--render_mask",      "-m",   is_flag=True,   help="to render masks?")
@click.option("--train_view",               is_flag=True,   help="to render with cam?")
@click.option("--fps",                      type=int,       default=30, help="fps of the rendered video")
@click.option("--video_len",                type=int,       default=1, help="time in seconds of the rendered video")
def main(path, video, namespace, epochs, aabb, n_frames, sigma_threshold, extract, colmap, gen_transform, train, render,
         render_mask, train_view, fps, video_len):
    if path:
        path = Path(path)
    elif video:
        path = Path(video).parent
    else:
        raise ValueError("you need to specify either 'video' or 'path'")
    if not path.exists():
        raise FileExistsError("path doesn't exist")

    if extract:
        extract_images_from_video(video, target_frames=n_frames)
    if colmap:
        run_colmap(path / "images")

    if gen_transform:
        generate_transforms(path, aabb)

    snapshot_file = path / "base.msgpack"
    if train:
        train_ngp(path, n_steps=epochs, snapshot_file=snapshot_file)

    if render:
        if not train_view:
            generate_camera_path_training_view(path, namespace)
        render_ngp(path, snapshot_file, sigma_threshold, namespace,
                   mode="shade", training_view=train_view, video_fps=fps, video_n_seconds=video_len)
        render_ngp(path, snapshot_file, sigma_threshold, namespace,
                   mode="depth", training_view=train_view, video_fps=fps, video_n_seconds=video_len)

    if render_mask:
        # first fetch aabb and then use depth maps to generate masks
        snapshot_file = path / "base_mask.msgpack"
        get_crop_box(path, snapshot_file, namespace)
        generate_masks(path, namespace)


if __name__ == '__main__':
    main()
