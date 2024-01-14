import os
import cv2
import click
import torch
from pathlib import Path
from typing import Dict
from robot_utils import console
from robot_utils.py.utils import load_dict_from_yaml
from robot_utils.cv.io.io_cv import load_rgb, write_binary_mask, write_rgb
from robot_utils.cv.io.io import image_to_video
from robot_utils.py.filesystem import get_ordered_files, validate_path, create_path, get_ordered_subdirs

from robot_vision.grounded_sam.grounded_sam import GroundedSAM
from robot_vision.grounded_sam.utils import get_indices_from_pred_dict
from robot_vision.utils.utils import get_default_checkpoints_path
from wrapper import SAMTracker


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--demo_path",        "-m",   type=str,       help="stereo or mono")
@click.option("--high_score_only",  "-h",   is_flag=True,   help="only detect highest score")
@click.option("--viz",              "-v",   is_flag=True,   help="viz the process")
@click.option("--video",            "-vi",  is_flag=True,   help="viz the process")
def main(demo_path: Path, high_score_only: bool = True, viz: bool = True, video: bool = True):
    run_sam_track_for_all_demos(demo_path, high_score_only, viz, video)


def run_sam_track_for_all_demos(
        demo_path: Path,
        high_score_only: bool = True,
        viz: bool = True,
        video: bool = True
):
    recording_list = get_ordered_subdirs(demo_path)
    for recording in recording_list:
        run_sam_track_for_each_recording(recording, high_score_only, viz, video)


def run_sam_track_for_each_recording(
        recording_path: Path,
        high_score_only: bool = True,
        viz: bool = True,
        video: bool = True
):
    recording_path = validate_path(recording_path, throw_error=True)[0]
    image_path = recording_path / "rgb"
    scene_graph_config = load_dict_from_yaml(recording_path / "scene_graph.yaml")
    run_sam_track_with_text_prompt(scene_graph_config["text_prompt"], image_path, high_score_only, viz, video)


def run_sam_track_with_text_prompt(
        text_prompt_dict: Dict[str, str],
        image_path: Path,
        high_score_only: bool = True,
        viz: bool = True,
        video: bool = True
):
    obj_list = text_prompt_dict.keys()
    text_prompt = " . ".join([text_prompt_dict[k] for k in obj_list])
    console.log(f"[bold green]the text prompt for objects {obj_list} is {text_prompt}")

    default_checkpoint_dino = get_default_checkpoints_path() / "groundingdino/groundingdino_swint_ogc.pth"
    default_checkpoint_sam = get_default_checkpoints_path() / "sam/sam_vit_h_4b8939.pth"
    model = GroundedSAM(
        text_prompt, highest_score_only=high_score_only,
        checkpoints_dino=os.environ.get("grounding_dino_MODEL_PATH", default_checkpoint_dino),
        checkpoints_sam=os.environ.get("segment_anything_MODEL_PATH", default_checkpoint_sam)
    )
    tracker = SAMTracker()

    for obj in obj_list:
        create_path(image_path.parent / f"mask/{obj}")
    create_path(image_path.parent / f"mask/all")

    demo_image_list_rgb = get_ordered_files(image_path)
    obj_idx = {}
    for image_id, image_file in enumerate(demo_image_list_rgb):
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or k == ord("q"):
            break

        img = load_rgb(image_file, bgr2rgb=True)
        if tracker.step == 0:
            pred_dict = model.run(img) if tracker.step == 0 else None
            if isinstance(pred_dict["masks"], torch.Tensor):
                pred_dict['masks'] = pred_dict['masks'].cpu().squeeze(1).numpy()

            for obj in obj_list:
                obj_idx[obj] = get_indices_from_pred_dict(pred_dict, text_prompt_dict[obj])[0]
        else:
            pred_dict = None

        images, masks = tracker.track(img, pred_dict, viz=True)
        for obj in obj_list:
            write_binary_mask(image_path.parent / f"mask/{obj}/{image_id:>06d}.png", masks[obj_idx[obj]])

        write_rgb(image_path.parent / f"mask/all/{image_id:>06d}.png", images, True)
        if viz:
            cv2.imshow("prediction", images)

    if video:
        image_to_video(image_path.parent / f"mask/all", image_path.parent / f"mask/all.mp4", "rgb", codec="MPEG")


if __name__ == "__main__":
    try:
        # main()
        p = "/media/gao/temp/fromJeff/data_from_xiaoshu/reproducable_skills/asym/pour_for_p2c/recordings/20230221_194735"
        run_sam_track_for_each_recording(Path(p))
    finally:
        cv2.destroyAllWindows()
