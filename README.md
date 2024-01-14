# Robot Vision

## Install dependencies

prepare the environment variables like so
```shell
###### robot-vision ########
export DEFAULT_CHECKPOINT_PATH=/absolute_path_to/checkpoints
export DEFAULT_DATASET_PATH=/absolute_path_to/datasets
```

Following the [instruction](https://github.com/wyngjf/robot-utils-pub) to install `robot-utils`.

Then, 
```shell
# if you need mediapipe
pip install -e .[mediapipe]

# otherwise 
pip install -e .
```

run `robot_vision_install`, and select the packages you need to install.
 
## Usage

See [Tutorials](./docs/tutorial.md)