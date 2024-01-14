import os
import click
from pathlib import Path
from robot_utils import console
from robot_utils.py.filesystem import create_path, validate_path
from robot_utils.pkg.install import install_deps, create_deps_template

from robot_vision.utils.utils import get_root_path, get_default_checkpoints_path


def get_install_dir():
    install_dir = os.environ.get("ROBOT_VISION_DEPS_INSTALL_DIR", None)
    if install_dir is None:
        install_dir = create_path(get_root_path().parent.parent / "deps/vision")
    console.log(f"[bold cyan]- Installing directory is: {install_dir}")
    return Path(install_dir)


def install():
    console.rule("installing deps for robot-vision package")
    package_path = get_root_path().parent
    console.log(f"[bold cyan]- current package: {package_path}")

    install_path = get_install_dir()
    config_path = validate_path(package_path / "robot_vision/meta/config", throw_error=True)[0]
    data_path = validate_path(get_default_checkpoints_path(), throw_error=True)[0]
    install_deps(package_path, config_path, install_path, data_path)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--name", "-n", type=str, help="the name of the dependency package (check the project name)")
def generate(name: str):
    from robot_utils.py.interact import ask_list
    console.rule("creating deps for robot-vision package")
    package_path = get_root_path().parent
    console.log(f"[bold cyan]- current package: {package_path}")

    config_path = validate_path(package_path / "robot_vision/meta/config", throw_error=True)[0]
    filename = config_path / f"{name}.yaml"
    console.log(f"[bold cyan]- writing to {filename}")
    if filename.is_file():
        console.log(f"[bold red]file exist, overwriting?")
        if ask_list("", ["yes", "no"]) == "no":
            return
    create_deps_template(filename)


if __name__ == "__main__":
    install()

