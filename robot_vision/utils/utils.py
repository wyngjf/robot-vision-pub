import os
from pathlib import Path


def get_root_path():
    return Path(__file__).parent.parent


def get_dataset_path():
    """
    return the dataset path.
    """
    return get_root_path() / "dataset"


def get_dcn_path():
    return get_root_path() / "dcn"


def get_mask_path():
    return get_root_path() / "mask"


def get_default_checkpoints_path():
    DEFAULT_CHECKPOINT_PATH = os.environ.get("DEFAULT_CHECKPOINT_PATH")
    if not DEFAULT_CHECKPOINT_PATH:
        from robot_utils.py.interact import ask_list
        if ask_list("if you are using LabPC, do you want to add the default "
                    "checkpoint path `/common/homes/staff/gao/dataset/train`?", ["yes", 'no']) == "yes":
            def add_to_bashrc():
                from robot_utils import console
                # Get the path to the user's home directory
                bashrc_path = Path.home() / ".bashrc"
                line_to_add = "export DEFAULT_CHECKPOINT_PATH=/common/homes/staff/gao/dataset/train\n"

                # Check if the line already exists in the .bashrc file
                with open(bashrc_path, 'r') as file:
                    lines = file.readlines()
                    if line_to_add in lines:
                        console.log("Line already exists in .bashrc")
                        return

                # Add the line to the .bashrc file
                with open(str(bashrc_path), 'a') as file:
                    file.write("###### robot-vision ########\n")
                    file.write(line_to_add)

            add_to_bashrc()

        raise EnvironmentError("missing DEFAULT_CHECKPOINT_PATH env variable, "
                               "please add it manually and try again")

    return Path(DEFAULT_CHECKPOINT_PATH)


def get_default_dataset_path():
    DEFAULT_DATASET_PATH = os.environ.get("DEFAULT_DATASET_PATH")
    if not DEFAULT_DATASET_PATH:
        from robot_utils.py.interact import ask_list
        if ask_list("if you are using LabPC, do you want to add the default "
                    "checkpoint path `/common/temp/from_jianfeng/dataset`?", ["yes", 'no']) == "yes":
            def add_to_bashrc():
                from robot_utils import console
                # Get the path to the user's home directory
                bashrc_path = Path.home() / ".bashrc"
                line_to_add = "export DEFAULT_DATASET_PATH=/common/temp/from_jianfeng/dataset\n"

                # Check if the line already exists in the .bashrc file
                with open(bashrc_path, 'r') as file:
                    lines = file.readlines()
                    if line_to_add in lines:
                        console.log("Line already exists in .bashrc")
                        return

                # Add the line to the .bashrc file
                with open(str(bashrc_path), 'a') as file:
                    file.write("###### robot-vision ########\n")
                    file.write(line_to_add)

            add_to_bashrc()

        raise EnvironmentError("missing DEFAULT_DATASET_PATH env variable, "
                               "please add it manually and try again")

    path = Path(DEFAULT_DATASET_PATH)
    if not path.is_dir():
        raise FileNotFoundError(f"the dataset path {path} does not exist, please check DEFAULT_DATASET_PATH env variable")
    return path

