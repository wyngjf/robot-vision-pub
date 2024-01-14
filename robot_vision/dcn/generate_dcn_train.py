import click
import copy
import random
import math
from pathlib import Path

import numpy as np
import torch
from robot_utils import console
from robot_utils.py.interact import ask_checkbox_with_all
from robot_utils.py.utils import save_to_yaml, get_datetime_string, load_dict_from_yaml
from robot_utils.py.filesystem import get_ordered_subdirs, get_ordered_files, create_path
from robot_utils.cv.io.io_cv import load_rgb
from robot_vision.dataset.dex_nerf.utils import get_intrinsics_from_transform
from robot_utils.cv.image.op_torch import compute_image_mean_and_std_dev
from robot_vision.dcn.dataset import PDCDataset


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--data_root",        "-d",   type=str,       help="the path to the data root of a specific object")
@click.option("--namespace",        "-n",   type=str,       help="the namespace to use")
@click.option("--train_root",       "-t",   type=str,       help="the root directory for training dcn models")
@click.option("--obj",              "-o",   type=str,       help="object name")
@click.option("--scale",            "-s",   type=float,     default=1.0, help="scale the images for faster training")
@click.option("--n_images",                 type=int,       default=500, help="number of images to compute mean/std")
def generate_dcn_train_cfg(data_root, namespace, train_root, obj, scale, n_images):
    rgb_ext = ".jpg"
    dep_ext = ".png"
    mask_ext = ".png"

    data_root = Path(data_root)
    if not data_root.exists():
        console.print(f"[red]{data_root} does not exist")
        exit(1)

    output_dir = create_path(Path(train_root, obj, get_datetime_string()))
    folder_list = get_ordered_subdirs(data_root)
    if len(folder_list) == 0:
        console.print(f"[bold red]No scene found under {data_root}")
        exit(1)

    # Step: 1) get number of images in all scenes
    _n_images = {}
    _num_images_total = 0
    for scene in folder_list:
        img_path = scene / namespace / "shade"
        n_images_in_scene = len(get_ordered_files(img_path, pattern=[rgb_ext]))
        _n_images[scene.stem] = n_images_in_scene
        _num_images_total += n_images_in_scene

    # Step: 2) ask user to select training and evaluation portion
    folder_list = [folder.stem for folder in folder_list]
    if input("choose training subset y/n? ") == "y":
        instruction = "'up/down' and 'space' to select, 'Enter' to exit"
        train = ask_checkbox_with_all(f"Choose training subset, {instruction}", folder_list)
        test = ask_checkbox_with_all(f"Choose testing subset, {instruction}", folder_list)
    else:
        train = folder_list
        test = folder_list

    # # Step: 3) compute mean and std of training portion
    # _image_filename_list = []
    # num_samples_per_scene = math.ceil(n_images / len(folder_list))
    # ic(folder_list, num_samples_per_scene)
    # for scene in folder_list:
    #     img_path = data_root / scene / namespace / "shade"
    #     n_images_in_scene = len(get_ordered_files(img_path, pattern=[rgb_ext]))
    #     ic(scene, n_images_in_scene)
    #     img_idx = torch.randperm(n_images_in_scene)[:min(num_samples_per_scene, n_images_in_scene)]
    #     for i in img_idx:
    #         _image_filename_list.append(img_path / f"{i:>04d}{rgb_ext}")
    #
    # mean, std = compute_image_mean_and_std_dev(_image_filename_list)
    # console.print(f"[blue]mean: {mean}, std: {std}")

    # Step: 4) read image information:
    rgb_file = data_root / folder_list[0] / namespace / f"shade/{0:04d}.jpg"
    if rgb_file.is_file():
        img = load_rgb(rgb_file)
        height, width = img.shape[:2]
    else:
        console.print(f"[bold red] sample image {rgb_file} doesn't exist, use default width 1280 and height 720")
        height, width = 720, 1280

    # Step: 5) read camera intrinsic
    transform_file = data_root / folder_list[0] / f"transforms.json"
    transform = load_dict_from_yaml(transform_file)
    intrinsic = get_intrinsics_from_transform(transform, scale)

    # Step: 5) write config to training root folder
    ic(rgb_ext, dep_ext, mask_ext)
    config = dict(
        data_root=str(data_root),
        namespace=namespace,
        obj=obj,
        rgb_ext=rgb_ext,
        dep_ext=dep_ext,
        mask_ext=mask_ext,
        scale=scale,
        width=width,
        height=height,
        # image_mean=mean.numpy().tolist(),
        # image_std=std.numpy().tolist(),
        image_mean=[],
        image_std=[],
        n_images_total=_num_images_total,
        n_images=_n_images,
        all=folder_list,
        train=copy.copy(train),
        eval=copy.copy(test),
        cam_intrinsic=intrinsic.tolist()
    )
    out_file = output_dir / f"data_config.yaml"
    console.print(f"[bold green]writing to {out_file}")
    console.print(f"[bold green]run the following command to train: \n")
    console.print(f"[bold cyan]python dcn/train.py -p {output_dir}\n\n")
    console.print(f"[bold green]and use the following to check the training results:\n")
    console.print(f"[bold cyan]python dcn/viz/dcn_heatmap.py -p {output_dir} -m eval\n\n")
    save_to_yaml(config, out_file, default_flow_style=False)


if __name__ == "__main__":
    generate_dcn_train_cfg()
