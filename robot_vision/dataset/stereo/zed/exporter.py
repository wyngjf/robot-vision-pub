import cv2
import enum
import click
import numpy as np
import pyzed.sl as sl
from pathlib import Path
from typing import Union
from rich.progress import Progress

from robot_utils import console
from robot_utils.py.filesystem import create_path


class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3


def export(filename: Union[str, Path], mode: int, tn_frames: int = -1, output_path: Union[str, Path] = None):
    """

    Args:
        filename: the absolute path to the .svo recording file
        mode: the mode of the extracted format.
        tn_frames: target number of frames, default -1 to extract all frames
        output_path: the output path of the images
    """
    svo_input_path = Path(filename)
    if output_path is None:
        output_path = create_path(svo_input_path.parent / "images")
    output_path = Path(output_path)

    output_as_video = mode in [0, 1]

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        console.log(f"[bold red]open camera status: {repr(err)}")
        zed.close()
        exit()

    # Get image size
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()

    video_writer = None
    if output_as_video:
        # Create video writer with MPEG-4 part 2 codec
        video_writer = cv2.VideoWriter(str(output_path),
                                       cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                       max(zed.get_camera_information().camera_configuration.fps, 25),
                                       (width_sbs, height))

        if not video_writer.isOpened():
            console.log("[bold red]OpenCV video writer cannot be opened. "
                        "Please check the .avi file path and write permissions.")
            zed.close()
            exit()

    rt_param = sl.RuntimeParameters()
    # rt_param.sensing_mode = sl.SENSING_MODE.FILL

    # Start SVO conversion to AVI/SEQUENCE
    n_frames = zed.get_svo_number_of_frames()
    if tn_frames <= 0:
        tn_frames = n_frames
    frame_idx = np.round(np.linspace(0, n_frames-1, tn_frames)).astype(int)
    with Progress() as p:
        p.console.rule("[bold blue]Converting SVO..., press Ctrl-C to quit")
        task = p.add_task("[blue]Converting", total=n_frames)
        i = 0
        while True:
            if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
                svo_position = zed.get_svo_position()

                # Retrieve SVO images
                zed.retrieve_image(left_image, sl.VIEW.LEFT)

                if mode == 2:
                    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                elif mode == 3 or mode == 1:
                    zed.retrieve_image(right_image, sl.VIEW.DEPTH)
                elif mode == 4:
                    zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

                if (svo_position-1) != frame_idx[i]:
                    p.update(task, advance=1)
                    continue

                i += 1
                if output_as_video:
                    # Copy the left image to the left side of SBS image
                    svo_image_sbs_rgba[0:height, 0:width, :] = left_image.get_data()

                    # Copy the right image to the right side of SBS image
                    svo_image_sbs_rgba[0:, width:, :] = right_image.get_data()

                    # Convert SVO image from RGBA to RGB
                    ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba, cv2.COLOR_RGBA2RGB)

                    # Write the RGB image in the video
                    video_writer.write(ocv_image_sbs_rgb)
                else:
                    # Generate file names
                    second_image_name = "right" if mode == 2 else "depth"
                    second_image_ext = ".png" if second_image_name == "depth" else ".jpg"
                    filename1 = output_path / f"left_{i:>06d}.jpg"
                    filename2 = output_path / f"{second_image_name}_{i:>06d}{second_image_ext}"

                    # Save Left images
                    cv2.imwrite(str(filename1), left_image.get_data())

                    if mode != 4:
                        # Save right images
                        cv2.imwrite(str(filename2), right_image.get_data())
                    else:
                        # Save depth images (convert to uint16)
                        cv2.imwrite(str(filename2), depth_image.get_data().astype(np.uint16))

                # Check if we have reached the end of the video
                p.update(task, advance=1)
                if svo_position >= (n_frames - 1):  # End of SVO
                    break

    if output_as_video:
        # Close the video writer
        video_writer.release()

    zed.close()
    console.rule("FINISH")
    return 0


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--filename",  "-f",   type=str,   help="absolute path to the recording xxx.svo")
@click.option("--mode",      "-m",   type=int,   help="mode of the exporter")
@click.option("--tn_frames", "-n",   type=int,   default=-1, help="target number of frames")
def main(filename, mode, tn_frames):
    export(filename, mode, tn_frames)


if __name__ == "__main__":
    main()
