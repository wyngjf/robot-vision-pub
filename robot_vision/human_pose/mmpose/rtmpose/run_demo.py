import click
import mmcv

from robot_utils import console
from robot_utils.cv.io.io_cv import load_rgb
from robot_utils.py.filesystem import create_path, validate_file
from robot_utils.serialize.dataclass import save_to_yaml
from robot_vision.human_pose.mmpose.rtmpose.rtmpose_wrapper import RTMPoseWrapper
from robot_vision.utils.utils import get_root_path


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--image_path", "-p", type=str, help="the absolute path to the training root folder of dcn")
def main(image_path):
    # image_path = Path(
    #     # "/media/gao/temp/fromJeff/stereo_recording/demo_handover_toolbox/20230221_161113/images/left_000061.png"
    #     # "/home/gao/projects/control/deps/vision/mmpose/output/000000.jpg"
    #     "/home/gao/dataset/kvil/demo/pour/pour_beer/recordings/20230803_112315_heavy_occlusion/kvil/rgb/000160.png"
    # )
    image_path = validate_file(image_path)[0]
    img = load_rgb(image_path, bgr2rgb=False)
    output_root = create_path(get_root_path().parent / "output/mmpose")

    model = RTMPoseWrapper()
    pred, img = model.detect_on_images(img, True)
    if not pred:
        console.log("[bold red]No human detected")
        exit()

    save_to_yaml(pred, output_root / "pred_instances.yaml")
    output_file = (output_root / image_path.name).resolve()
    mmcv.imwrite(img, str(output_file))

    model.get_meta_info(output_root / "meta.json")


if __name__ == '__main__':
    main()
