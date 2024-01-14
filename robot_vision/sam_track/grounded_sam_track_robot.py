import os
import cv2
import click
from robot_vision.grounded_sam.grounded_sam import GroundedSAM
from robot_vision.utils.utils import get_default_checkpoints_path

from armarx_control.robots.common import Robot, cfg
from wrapper import SAMTracker


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--text_prompt",      "-t",   type=str,       help="the text prompt")
@click.option("--cam_mode",         "-m",   type=str,       help="stereo or mono")
@click.option("--high_score_only",  "-h",   is_flag=True,   help="only detect highest score")
def main1(text_prompt, cam_mode, high_score_only):
    robot_cfg = cfg.RobotConfig()
    if cam_mode == "stereo":
        robot_cfg.stereo = cfg.StereoCameraConfig()
    else:
        robot_cfg.mono = cfg.MonocularCameraConfig()
    robot = Robot(robot_cfg)

    default_checkpoint_dino = get_default_checkpoints_path() / "groundingdino/groundingdino_swint_ogc.pth"
    default_checkpoint_sam = get_default_checkpoints_path() / "sam/sam_vit_h_4b8939.pth"
    model = GroundedSAM(
        text_prompt, highest_score_only=high_score_only,
        checkpoints_dino=os.environ.get("grounding_dino_MODEL_PATH", default_checkpoint_dino),
        checkpoints_sam=os.environ.get("segment_anything_MODEL_PATH", default_checkpoint_sam)
    )
    tracker = SAMTracker()

    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or k == ord("q"):
            break

        if cam_mode == "stereo":
            images, info = robot.get_stereo_images()
            img = images[0]
        else:
            img, depth, info = robot.get_mono_images(True)

        pred_dict = model.run(img) if tracker.step == 0 else None
        images, masks = tracker.track(img, pred_dict, viz=True)

        cv2.imshow("prediction", images)


# @click.command(context_settings=dict(help_option_names=['-h', '--help']))
# @click.option("--text_prompt",      "-t",   type=str,       help="the text prompt")
# @click.option("--cam_mode",         "-m",   type=str,       help="stereo or mono")
# @click.option("--high_score_only",  "-h",   is_flag=True,   help="only detect highest score")
# def main(text_prompt, cam_mode, high_score_only):
#     robot_cfg = cfg.RobotConfig()
#     if cam_mode == "stereo":
#         robot_cfg.stereo = cfg.StereoCameraConfig()
#     else:
#         robot_cfg.mono = cfg.MonocularCameraConfig()
#     robot = Robot(robot_cfg)
#     # from robot_utils.py.filesystem import get_ordered_files
#     # from robot_utils.cv.io.io_cv import load_rgb, load_depth
#     #
#     # path = Path("/common/homes/all/gao/dataset/demo/pour_water_11/recordings/2023-01-18-19-15-23")
#     # rgb_files = get_ordered_files(path / "rgb", pattern=[".jpg"])
#     # depth_files = get_ordered_files(path / "depth", pattern=[".png"])
#     # ic(rgb_files)
#     # ic(depth_files)
#     # if len(rgb_files) == 0:
#     #     raise RuntimeError("no file founded")
#
#     # def get_images(idx: int):
#     #     rgb = load_rgb(rgb_files[idx])
#     #     depth = load_depth(depth_files[idx], to_meter=True)
#     #     console.log(f"getting image in step {idx}: {rgb_files[idx]}, rgb: {rgb.shape}, depth: {depth.shape}")
#     #     return rgb, depth
#
#     default_checkpoint_dino = "/common/homes/staff/gao/code/deps/grounding_dino/checkpoints/groundingdino_swint_ogc.pth"
#     default_checkpoint_sam = "/common/homes/staff/gao/code/deps/segment_anything/checkpoints/sam_vit_h_4b8939.pth"
#     model = GroundedSAM(
#         text_prompt, highest_score_only=high_score_only,
#         checkpoints_dino=os.environ.get("grounding_dino_MODEL_PATH", default_checkpoint_dino),
#         checkpoints_sam=os.environ.get("segment_anything_MODEL_PATH", default_checkpoint_sam)
#     )
#
#     segtracker = SegTracker(segtracker_args, sam_args, aot_args)
#     segtracker.restart_tracker()
#
#     mask_last_frame = None
#     mask_this_frame = None
#     step = 0
#
#     def viz(step: int, visualizer=None):
#         if cam_mode == "stereo":
#             images, info = robot.get_stereo_images()
#             img = images[0]
#         else:
#             img, depth, info = robot.get_mono_images(True)
#         # img, depth = get_images(step)
#
#         if step == 0:
#             pred_dict = model.run(img)
#             image = visualize_pred(copy.deepcopy(img), pred_dict)
#             # for mask in pred_dict["masks"]:
#             #     ic(image.shape, mask.shape)
#             #     segtracker.add_reference(copy.deepcopy(img), mask.squeeze().cpu().numpy())
#             ic(pred_dict["masks"].shape)
#             segtracker.add_masks(copy.deepcopy(img), pred_dict["masks"].squeeze(1).cpu().numpy())
#
#         else:
#             track_mask = segtracker.track(frame=img, update_memory=True)
#             ic(track_mask.shape, np.unique(track_mask))
#             image = overlay_masks_on_image(img, [track_mask])
#
#         # mask_last_frame = mask_this_frame
#         # mask_this_frame = segtracker.track(frame=img, update_memory=True)
#
#         cv2.imshow("prediction", image)
#         return visualizer
#
#     v = viz(0)
#
#     while True:
#         step += 1
#         k = cv2.waitKey(10) & 0xFF
#         if k == 27 or k == ord("q"):
#         # if k == 27 or k == ord("q") or step == len(rgb_files):
#             break
#
#         viz(step, v)


if __name__ == "__main__":
    try:
        main1()
    finally:
        cv2.destroyAllWindows()
