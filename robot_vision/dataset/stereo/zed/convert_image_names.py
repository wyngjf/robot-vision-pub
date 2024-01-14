import click
from typing import List
from pathlib import Path
from robot_utils.py.filesystem import get_ordered_files, rename_path, get_ordered_subdirs
from robot_utils.py.interact import ask_checkbox


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--working_path", "-p", type=str, help="the path to the working directory a K-VIL object")
def convert(working_path):
    working_path = Path(working_path)
    all_scenes = ask_checkbox("choose scenes to proceed", get_ordered_subdirs(working_path))  # type: List[Path]
    for path in all_scenes:
        path = Path(path)
        patterns = ["left_", "right_"]
        postfix = ["_l", "_r"]
        index = [5, 6]
        for i in range(2):
            img_list = get_ordered_files(path / "images", pattern=[patterns[i]])
            for f in img_list:
                new_name = f.parent / f"{f.stem[index[i]:]}{postfix[i]}.png"
                rename_path(f, new_name)


if __name__ == "__main__":
    convert()

