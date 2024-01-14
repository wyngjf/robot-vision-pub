import os
from pathlib import Path
from typing import Union
from robot_utils.py.filesystem import validate_path
from robot_utils import console
from robot_utils.py.system import run

NGP_PATH = os.environ.get("NGP_PATH")
if not NGP_PATH:
    console.rule("[bold red]missing DEX_NGP_PATH env variable")
    exit(1)


def viz_testbed(
        scene_dir: Union[str, Path],
        snapshot_file: Union[str, Path] = None,
):
    scene_dir, _ = validate_path(scene_dir, throw_error=True)
    snapshot_file = scene_dir / 'base.msgpack' if snapshot_file is None else snapshot_file
    cmd = f"./build/testbed --scene {str(scene_dir)} --no-train --snapshot {str(snapshot_file)}"
    run(cmd, working_dir=NGP_PATH)


if __name__ == "__main__":
    scene = "/media/gao/dataset/kvil/kettle/2022-12-29-17-14-51"
    viz_testbed(scene)
