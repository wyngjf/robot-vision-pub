import os
import cv2
import click
from grounded_sam import GroundedSAM, GroundedSAMConfig, visualize_pred
from robot_vision.utils.utils import get_default_checkpoints_path
from armarx_control.robots.common import Robot, cfg


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--text_prompt",      "-t",   type=str,       help="the text prompt")
@click.option("--cam_mode",         "-m",   type=str,       help="stereo or mono")
@click.option("--high_score_only",  "-h",   is_flag=True,   help="only detect highest score")
@click.option("--hq_token_only",    "-hq",  is_flag=True,   help="use the high-quality SAM")
def main(text_prompt, cam_mode, high_score_only, hq_token_only):
    robot_cfg = cfg.RobotConfig()
    robot_cfg.stereo = cfg.StereoCameraConfig() if cam_mode == "stereo" else cfg.MonocularCameraConfig()
    robot = Robot(robot_cfg)

    default_checkpoint_dino = get_default_checkpoints_path() / "groundingdino/groundingdino_swint_ogc.pth"
    default_checkpoint_sam = get_default_checkpoints_path() / "sam/sam_vit_h_4b8939.pth"

    config = GroundedSAMConfig()
    config.flag_highest_score_only = high_score_only
    config.sam.flag_sam_hq = hq_token_only
    config.sam.checkpoints = os.environ.get("segment_anything_MODEL_PATH", default_checkpoint_sam)
    config.llm.checkpoints = os.environ.get("grounding_dino_MODEL_PATH", default_checkpoint_dino)
    model = GroundedSAM(config)

    def viz(visualizer=None):
        if cam_mode == "stereo":
            images, info = robot.get_stereo_images()
            img = images[0]
        else:
            img, depth, info = robot.get_mono_images(True)

        pred_dict = model.run(img, text_prompt)
        image = visualize_pred(img, pred_dict)

        cv2.imshow("prediction", image)
        return visualizer

    v = viz()

    while True:
        k = cv2.waitKey(200) & 0xFF
        if k == 27 or k == ord("q"):
            break

        viz(v)


if __name__ == "__main__":
    try:
        main()
    finally:
        cv2.destroyAllWindows()
