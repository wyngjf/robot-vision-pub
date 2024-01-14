import pyzed.sl as sl
import ogl_viewer.viewer as gl
from time import time

from robot_utils.py.filesystem import get_home, create_path
from robot_utils.py.utils import get_datetime_string, save_to_yaml, dump_data_to_yaml
from robot_utils import console

from robot_vision.dataset.stereo.zed.type import ZEDCameraParam, ZEDCalibration


def record():
    console.rule("[bold blue]ZED recording")
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        console.log(f"[bold red]open camera status: {repr(status)}")
        exit(1)

    path = create_path(get_home() / "dataset/raw" / get_datetime_string())
    console.log(f"Writing to {path}")
    recording_param = sl.RecordingParameters(str(path / "recording.svo"), sl.SVO_COMPRESSION_MODE.LOSSLESS)
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        console.log(f"[bold red]set recording param status: {repr(err)}")
        exit(1)

    calibration_params_raw = cam.get_camera_information().calibration_parameters_raw
    calibration_params = cam.get_camera_information().camera_configuration.calibration_parameters
    param_l = calibration_params.left_cam
    param_r = calibration_params.right_cam
    calibrate_param = ZEDCameraParam()
    cam_l = ZEDCalibration()
    cam_r = ZEDCalibration()
    cam_l.fx = param_l.fx
    cam_l.fy = param_l.fy
    cam_l.cx = param_l.cx
    cam_l.cy = param_l.cy

    cam_r.fx = param_r.fx
    cam_r.fy = param_r.fy
    cam_r.cx = param_r.cx
    cam_r.cy = param_r.cy
    calibrate_param.left = cam_l
    calibrate_param.right = cam_r
    calibrate_param.image_size = [calibration_params_raw.left_cam.image_size.width,
                                  calibration_params_raw.left_cam.image_size.height]
    calibrate_param.rotation = calibration_params.R.tolist()
    calibrate_param.translation = calibration_params.T.tolist()
    distortion_key = ["k1", "k2", "k3", "p1", "p2"]
    for key, v_l, v_r in zip(
            distortion_key,
            calibration_params_raw.left_cam.disto.tolist(),
            calibration_params_raw.right_cam.disto.tolist()
    ):
        setattr(calibrate_param.left, key, v_l)
        setattr(calibrate_param.right, key, v_r)

    dump_data_to_yaml(ZEDCameraParam, calibrate_param, str(path / "param.yaml"))
    console.log(f"[bold green]saving camera config to {path / 'param.yaml'}")

    runtime = sl.RuntimeParameters()
    console.log("[bold green]SVO is Recording, use Ctrl-C or 'q' to stop.")

    # OpenGL Viewer
    camera_info = cam.get_camera_information()
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam)

    image_l = sl.Mat()
    image_r = sl.Mat()

    frames_recorded = 0
    start = time()
    while viewer.is_available():
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image_l, sl.VIEW.LEFT)
            # cam.retrieve_image(image_r, sl.VIEW.RIGHT)
            frames_recorded += 1
            viewer.update_view(image_l)

    console.log(f"[bold yellow]In total: {frames_recorded} images for each view, duration {time() - start}")
    viewer.exit()
    image_l.free(sl.MEM.CPU)
    image_r.free(sl.MEM.CPU)
    cam.close()
    console.rule("[bold blue]FINISH")


if __name__ == "__main__":
    record()
