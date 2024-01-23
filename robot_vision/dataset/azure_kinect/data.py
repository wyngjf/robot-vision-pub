import os

import click
import time
import datetime
import logging
import open3d as o3d
from os.path import join, isdir
from marshmallow_dataclass import dataclass, class_schema
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

from robot_utils.py.utils import load_dataclass
from robot_utils.py.interact import ask_checkbox
from robot_utils.py.filesystem import create_path, get_ordered_subdirs, copy2
from robot_vision.utils.utils import get_dataset_path, get_root_path
from robot_vision.dataset.data_utils.viz.o3d_viz import draw_geometries_flip


@dataclass
class ReconstructConfig:
    max_depth:                              float = 3.0
    min_depth:                              float = 0.2
    voxel_size:                             float = 0.05

    icp_method:                             str = "color"
    lambda_geometric:                       float = 0.968

    max_depth_diff:                         float = 0.07
    tsdf_cubic_size:                        float = 3.0
    global_registration:                    str = "ransac"

    n_frames_per_fragment:                  int = 60
    n_keyframes_per_n_frame:                int = 3
    python_multi_threading:                 bool = True
    preference_loop_closure_odometry:       float = 0.1
    preference_loop_closure_registration:   float = 5.0

    # # folder contains data, default is the working space root dir
    # PDC_ROOT_DIR = os.getcwd()
    #
    # PDC_DATA_DIR: str = join(PDC_ROOT_DIR, "data")
    # LOGS_PROTO_DIR: str = join(PDC_ROOT_DIR, "data/logs_proto")

    # generate two folders: data/fragments and data/scene
    folder_fragment:                    str = "fragments/"
    fragment_posegraph:                 str = "fragments/fragment_%03d.json"
    fragment_posegraph_optimized:       str = "fragments/fragment_optimized_%03d.json"  # make_fragments.py posegraph of the imgs
    fragment_pointcloud:                str = "fragments/fragment_%03d.ply"

    folder_scene:                       str = "scene/"
    global_posegraph:                   str = "scene/global_registration.json"
    global_posegraph_optimized:         str = "scene/global_registration_optimized.json"  # register_fragments.py posegraph of the fragments

    refined_posegraph:                  str = "scene/refined_registration.json"  # refine_registration.py posegraph of the fragments refined
    refined_posegraph_optimized:        str = "scene/refined_registration_optimized.json"

    global_mesh:                        str = "scene/integrated.ply"  # integrate_scene.py
    global_traj:                        str = "scene/trajectory.log"

    # `slac` and `slac_integrate` related parameters.
    # `voxel_size` and `min_depth` paramters from previous section,
    # are also used in `slac` and `slac_integrate`.
    max_iterations:                     int = 5
    depth_scale:                        int = 1000  # TODO type
    sdf_trunc:                          float = 0.04
    block_count:                        int = 40000
    distance_threshold:                 float = 0.07
    fitness_threshold:                  float = 0.3
    regularizer_weight:                 int = 1
    method:                             str = "slac" # rigid
    device:                             str = "CUDA:0"  # CUDA:0 CPU:0
    save_output_as:                     str = "pointcloud"
    folder_slac:                        str = "slac/"
    template_optimized_posegraph_slac:  str = "optimized_posegraph_slac.json"
    subfolder_slac:                     str = "slac/%0.3f/" % voxel_size

    debug_mode:                         bool = False


def record(obj: str = "OBJECT", info: str = ""):
    from robot_utils.sensor.azure_kinect.recorder import RecorderWithCallback

    start_time = time.time()
    recorder = RecorderWithCallback(
        azure_config=str(get_dataset_path() / "config/azure_kinect_default_config.json"),
        recording_path=str(create_path(get_root_path() / "data" / obj)),
        folder_name='{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now()),
        obj=obj, info=info,
        align_depth_to_color=False,
        sensor_index=0)
    recorder.run()
    logging.info(f"recording finished in {time.time() - start_time}")


def extract(obj, crop=False, force=False):  # TODO fix this
    # from pathlib import Path
    # from robot_utils.py.filesystem import get_ordered_files
    # from robot_utils.cv.io.io import extract_images_from_video
    # path = Path("/home/jianfeng/projects/robot-vision/robot_vision/data/OBJECT")
    # all_scenes = ask_checkbox("choose scenes to proceed", get_ordered_subdirs(path))
    # for p in all_scenes:
    #     video = get_ordered_files(p, pattern=[".mkv"])[0]
    #     extract_images_from_video(video, target_frames=10)
    # exit()

    from robot_utils.sensor.azure_kinect.mkv_reader import ReaderWithCallback

    src_dir = join(get_data_path(), "raw", obj)
    folder_list = get_ordered_subdirs(src_dir, to_str=True)

    # def process(folder_name, crop):
    #     working_dir = folder_name.replace("raw", "processed")
    #     if not isdir(working_dir):
    #         create_path(working_dir)
    #         reader = ReaderWithCallback(mkv_dir=folder_name, working_dir=working_dir)
    #         reader.run(crop)
    #
    # pool = Pool(os.cpu_count() - 1)
    # pool.map(partial(process, crop=crop), folder_list)

    for folder in folder_list:
        working_dir = folder.replace("raw", "processed")
        if force or not isdir(working_dir):
            create_path(working_dir, force)
            reader = ReaderWithCallback(mkv_dir=folder, output_path=working_dir)
            reader.run(crop)


def reconstruct(obj, fragment, register, refine, integrate):
    import robot_vision.dataset.reconstruction.make_fragments as make_fragments
    import robot_vision.dataset.reconstruction.register_fragments as register_fragments
    import robot_vision.dataset.reconstruction.refine_registration as refine_registration
    import robot_vision.dataset.reconstruction.integrate_scene as integrate_scene

    rec_config_file = join(get_vil_path("datasets"), "config/reconstruction.yaml")
    rec_config = load_dataclass(ReconstructConfig, rec_config_file)
    config = class_schema(ReconstructConfig)().dump(rec_config)

    src_dir = join(get_data_path(), "processed", obj)
    folder_list = get_ordered_subdirs(src_dir, to_str=True)
    for working_dir in folder_list:
        copy2(rec_config_file, join(working_dir, "reconstruction.yaml"))
        config['path_intrinsic'] = join(working_dir, "images/camera_azurekinect.json")
        config['path_dataset'] = working_dir

        if os.path.isdir(os.path.join(working_dir, "fragments")):
            continue

        if fragment:
            make_fragments.run(config)

        if register:
            register_fragments.run(config)

        if refine:
            refine_registration.run(config)

        if integrate:
            integrate_scene.run(config)


def show_mesh(obj):
    src_dir = join(get_data_path(), "processed", obj)
    folder_list = get_ordered_subdirs(src_dir, to_str=True)
    folder_list = ask_checkbox("select folders to visualize: ", folder_list)

    if len(folder_list) == 0:
        logging.warning("you didn't choose any folder")

    for folder in folder_list:
        mesh1 = o3d.io.read_triangle_mesh(join(folder, "scene/integrated.ply"))
        draw_geometries_flip([mesh1])


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--obj",  "-o",   type=str,   default="OBJECT",   help="the object category")
@click.option("--info", "-i",   type=str,   default="",         help="additional information for recording")
@click.option("--rec",  "-r",   is_flag=True,                   help="flag to enable recording")
@click.option("--ext",  "-e",   is_flag=True,                   help="flag to extract all recordings in object folder")
@click.option("--rc",           is_flag=True,                   help="flag to enable reconstruction")
@click.option("--crop",         is_flag=True,                   help="flag to enable reconstruction")
@click.option("--show",         is_flag=True,                   help="flag to show the chosen reconstruction results")
@click.option("--fragment",     default=True,                   help="flag to make fragment")
@click.option("--register",     default=True,                   help="flag to ")
@click.option("--refine",       default=True,                   help="flag to ")
@click.option("--integrate",    default=True,                   help="flag to ")
@click.option("--force",        is_flag=True,                   help="force redo the step even if the dir exists")
def main(obj, info, rec, ext, rc, crop, show, fragment, register, refine, integrate, force):
    if rec:
        record(obj, info)

    if ext:
        extract(obj, crop=crop, force=force)

    if rc:
        reconstruct(obj, fragment, register, refine, integrate)

    if show:
        show_mesh(obj)


if __name__ == "__main__":
    main()
