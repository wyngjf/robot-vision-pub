import click

from robot_utils.py.filesystem import get_ordered_subdirs, validate_path
from robot_utils.py.interact import ask_checkbox

from robot_vision.dataset.stereo.zed.exporter import export


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",         "-p",   type=str,   help="abs path to the recordings")
@click.option("--tn_frames",    "-n",   type=int,   default=-1, help="target number of frames")
def main(path, tn_frames):
    path, _ = validate_path(path, throw_error=True)
    folders = ask_checkbox("select folders to proceed", get_ordered_subdirs(path))
    for folder in folders:
        export(str(folder / "recording.svo"), mode=2, tn_frames=tn_frames)


if __name__ == "__main__":
    main()
