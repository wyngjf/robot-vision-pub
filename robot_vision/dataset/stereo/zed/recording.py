import cv2
import numpy as np
import pyzed.sl as sl

from robot_utils import console
from robot_utils.py.filesystem import get_home, create_path
from robot_utils.py.utils import save_to_yaml, get_datetime_string


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    param_l = calibration_params.left_cam
    param_r = calibration_params.right_cam
    calibrate_param = dict(
        left=dict(
            fx=param_l.fx,
            fy=param_l.fy,
            cx=param_l.cx,
            cy=param_l.cy,
            # k1=param_l.k1,
            # k2=param_l.k2,
            # k3=param_l.k3,
            # p1=param_l.p1,
            # p2=param_l.p2,
        ),
        right=dict(
            fx=param_r.fx,
            fy=param_r.fy,
            cx=param_r.cx,
            cy=param_r.cy,
            # k1=param_r.k1,
            # k2=param_r.k2,
            # k3=param_r.k3,
            # p1=param_r.p1,
            # p2=param_r.p2,
        ),
        rotation=calibration_params.R.tolist(),
        translation=calibration_params.T.tolist(),
        # stereo_transform=calibration_params.stereo_transform.get().tolist()
    )

    record_path = create_path(get_home() / "dataset/raw" / get_datetime_string())
    console.log(f"writing to {record_path}")

    # Capture 50 frames and stop
    i = 0
    image_l = sl.Mat()
    image_r = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    save_to_yaml(calibrate_param, str(record_path / "param.yaml"))

    while i < 50:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image_l, sl.VIEW.LEFT)
            zed.retrieve_image(image_r, sl.VIEW.RIGHT)
            cv2.imwrite(str(record_path / f"{i:>04d}_l.png"), image_l.get_data())
            cv2.imwrite(str(record_path / f"{i:>04d}_r.png"), image_r.get_data())
            image = np.concatenate((image_l.get_data(), image_r.get_data()), axis=1)
            cv2.imshow("stereo", image_l.get_data())

            # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
            # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image_l.get_width(), image_l.get_height(),
                  # timestamp.get_milliseconds()))
            i = i + 1

    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()