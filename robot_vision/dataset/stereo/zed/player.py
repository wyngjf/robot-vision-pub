import cv2
import click
import pyzed.sl as sl
from robot_utils import console


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--filename",  "-f",   type=str,   help="absolute path to the recording xxx.svo")
def main(filename):
    console.rule("[bold blue]Play video")
    console.log(f"[bold blue]Reading SVO file: {filename}")

    input_type = sl.InputType()
    input_type.set_from_svo_file(filename)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        console.log(f"[bold red]open camera status: {repr(status)}")
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    key = ''
    console.log("[bold green]Press 'q' to quit")
    n_frames = cam.get_svo_number_of_frames()
    i = 0
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat)
            i += 1
            if i == n_frames:
                break
        cv2.imshow("ZED Recordings", mat.get_data())
        key = cv2.waitKey(1)
    cv2.destroyAllWindows()

    cam.close()
    console.rule("[bold blue]FINISH")


if __name__ == "__main__":
    main()
