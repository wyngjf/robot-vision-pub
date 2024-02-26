# Dex-NeRF

## Install dependencies

Install distribution package `colmap`
```shell
sudo apt install colmap
```
or build it from source:

> make sure you are not in a conda virtual environment, otherwise you may see compiling errors about "tiffxxx". 

Install dependencies:
```shell
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
```

GoogleTest
```shell
git clone https://github.com/google/googletest.git -b v1.13.0
cd googletest                 # Main directory of the cloned repository.
mkdir build                   # Create a directory to hold the build output.
cd build
cmake .. -DBUILD_GMOCK=OFF    # Generate native build scripts for GoogleTest.
make -j30
sudo make install             # Install in /usr/local/ by default
```

install ceres
```shell
mkdir -p $HOME/opt
cd $HOME/opt
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$(pwd)/install
make -j 30
make install
```

install colmap
```shell
cd $HOME/opt
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout dev
mkdir build
cd build
cmake .. -DCeres_DIR=$HOME/opt/ceres-solver/build/install/lib/cmake/Ceres/ -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DCMAKE_CUDA_ARCHITECTURES=75
make -j 30
make install
echo "export PATH=$HOME/opt/colmap/build/install/bin:$PATH" >> $HOME/.bashrc
source $HOME/.bashrc
```

## Setup Instant NGP
Setup [Dex Nerf](https://github.com/salykovaa/instant-DexNerf), which is basically a minor modification of NVIDIA's instant ngp. 
In order to build the cython binding that corresponds to the python version you are using, make sure to activate your virtual 
environment before compiling.

```shell
pip install cmake --upgrade
conda activate ENV_NAME
```

then 

```shell
git clone --recursive https://github.com/wyngjf/instant-ngp.git
cd instant-ngp
# the default branch should be origin/dex_nerf
git submodule  update --init --recursive
cmake . -B build_py310
cmake --build build_py310 --config RelWithDebInfo -j8
echo "export NGP_PATH=$(pwd)" >> $HOME/.bashrc
source $HOME/.bashrc
```


## Use Instant NGP
Once compiled, can use the scripts in `robot_vision/dataset/dex_nerf` to deal with your dataset.

1. Extraction of images, specify the path to your video and the expected number of frames to be extracted.
   ```shell
   cd PATH_TO_ROBOT_VISION/robot_vision/dataset/dex_nerf
   python main.py -video PATH_TO_VIDEO --extract --n_frames 100
   # e.g.
   python main.py -v /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36/2022-11-28-11-25-36.mkv -x -f 100
   ```
2. Run colmap and estimate camera pose, specify path to the scene folder
   ```shell
   python main.py --path PATH_TO_SCENE --colmap
   # e.g.
   python main.py --p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -c
   ```
3. generate `transforms.json`, which contains camera parameters, scene scaling and translation parameters, as well as image paths and camera poses. 
   you need to specify the scaling factor `aabb`. More details see instant ngp docs. Note that, colmap may fail to include several 
   images, therefore, the indices may be not continuous. We check if the number of the included images, if this number
   is larger than 50, then we can still continue and observe the training results of NGP, you'll need to decide whether 
   you need to run colmap again.
   ```shell
   python main.py --path PATH_TO_SCENE --gen_transform --aabb 8
   # e.g.
   python main.py --p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -g -ab 8
   ```
4. train instant-ngp, specify the training iterations `epochs`.
   ```shell
   python main.py --path PATH_TO_SCENE --train --epochs 5000 
   # e.g.
   python main.py --p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -t -e 5000
   ```
5. generate camera path for rendering into a `namespace`. Either generate from training view trajectory, or manually create a path in the ngp GUI
6. rendering a sequence of images and corresponding camera pose (depth map, rgb image, camera pose)
   ```shell
   # use the same view and generate the same number of images as the extracted images
   python main.py --path PATH_TO_SCENE --render --train_view --namespace train_view
   # otherwise, enable interpolation of camera poses along the training views to generate more images, but still following the training view.
   python main.py --path PATH_TO_SCENE --render --fps 30 --video_len 10 --namespace interpolate_train_view
   # To use manually designed cam path. You can adapt the main.py for this purpose
 
   # e.g.
   python main.py --p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -r --fps 30 --video_len 10 -n interpolate_train_view
   ```
   the images will be rendered into `PATH_TO_ROBOT_VISION/namespace` folder.
7. generate masks
   In order to generate mask, you need to manually crop the scene, it cannot be automated at the moment. You can load the scene in instant-ngp and save a snapshot file to the scene
   root folder with name `base_mask.msgpack`. This step will generate a yaml file `aabb.yaml` in the `kvil` folder.
   
   ```shell
   cd $NGP_PATH
   ./build/testbed --scene SCENE_PATH --no-train --snapshot SCENE_PATH/base.msgpack
   ```
   Then you can run the following scripts.
   ```shell
   python main.py --path PATH_TO_SCENE --namespace train_view --render_mask
   # e.g.
   python main.py --p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -n train_view -m
   ```
   
To run all at once for a specific scene, e.g.
```shell
python main.py -v SCENE_PATH/xxxx.mkv -n kvil -x -f 100 -c -g -ab 8 -t -r --train_view
python main.py -v SCENE_PATH/xxxx.mkv -n kvil -x -f 100 -c -g -ab 8 -t -r --fps 20 --video_len 20
python main.py -p SCENE_PATH -n kvil -m
# e.g.
python main.py -v /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36/2022-11-28-11-25-36.mkv -n kvil -x -f 100 -c -g -ab 8 -t -r --train_view
python main.py -v /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36/2022-11-28-11-25-36.mkv -n kvil -x -f 100 -c -g -ab 8 -t -r  --fps 20 --video_len 20
python main.py -p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -n kvil -r --fps 20 --video_len 20
python main.py -p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -n kvil -m
```

all run all steps for several scenes at once, e.g.
```shell
# extract, run colmap, generate transforms, train nerf, render
python run_all.py -p PATH_TO_OBJ_FOLDER -n kvil -x -f 100 -c -g -ab 8 -t -r --fps 20 --video_len 20
# visualize the trained model and crop the scene to save snapshot for mask rendering
python run_all.py -p PATH_TO_OBJ_FOLDER -n kvil -v
# with the saved 'base_mask.msgpack`, you can render masks now
python run_all.py -p PATH_TO_OBJ_FOLDER -n kvil -m
# render with rotated cameras, the settings of fps and video_len must be the same as above, you need to select the folders
# that rendered without rotated cameras, new folders will be created to store the rotated scenes
python run_all.py -p PATH_TO_OBJ_FOLDER -n kvil -r --fps 20 --video_len 20 --rand_rot --rand_range 0 90 -m

e.g. 
python run_all.py -p /common/homes/staff/gao/dataset/kvil/brush -n kvil -x -f 100 -c -g -ab 8 -t -r --fps 20 --video_len 20
python run_all.py -p /common/homes/staff/gao/dataset/kvil/brush -n kvil -v
python run_all.py -p /common/homes/staff/gao/dataset/kvil/brush -n kvil -m
```

## File structure

The folder structure of your `DATA_ROOT` will look like this after rendering step finishes
```shell
DATA_ROOT
- colmap                        # the Colmap results
  - colmap_sparse
  - colmap_text
  - colmap.db
- images						# extracted images for training [--extract, -e]
- namespace1                    # the results for a user-defined namespace
	- shade						# (rendered rgb images) [-r]
	- shade.mp4			        # rgb video
	- depth 					# (rendered depth images) [-r]
	- depth.mp4			        # depth video
	- mask						# (rendered mask images) [-m]
	- positions					# (rendered position images)
	- positions.mp4		        # positions video
	
	- base_cam.json				# a path of camera poses for rendering
	- intrinsics.yaml			# camera intrinsic matrix
	
	- pose_data.yaml			# the camera pose of each rendered image along the camera path
	- crop_box.yaml			    # include crop_box, crop_box_corners, and aabb_rotation
	- aabb.yaml			        # include lower, upper, and rotation param

- namespace2
- namespace3 ...
- 2021-12-13-16-44-24.mkv		# raw video recording

- base_mask.msgpack				# snapshot of the NGP Nerf model cropped to the object (manually done, before --mask step)
- base.msgpack					# snapshot of the NGP Nerf model of the entire scene
- base.obj						# generated mesh object (manually done) in [-v] step

- info.yaml                     # (optional) information during the recording
- transforms.json				# camera poses, camera params from colmap2nerf [--generate, -g] step
```


## Visualize correspondences

To examine the correspondences in a scene under a namespace, do:
```shell
# Run all with -vc flag to visualize the correspondence
python run_all.py -p PATH_TO_OBJ_FOLDER -n kvil -vc 
# also visualize non-correspondence
python run_all.py -p PATH_TO_OBJ_FOLDER -n kvil -vc -vcn

# or only visualize each individual scene 
python viz_correspondence.py --path PATH_TO_SCENE --namespace train_view 
python viz_correspondence.py -p /media/gao/dataset/kvil/dustpan/2022-11-28-11-25-36 -n train_view 
```
Hang your mouse over the first/left rgb image, the yellow dot appears and  indicates the correspondences between two different view points. See
Nerf-supervision for more details.

## Troubleshoot

if you see something like
```shell
$ ngp -h
Traceback (most recent call last):
  File "/home/gao/opt/anaconda3/envs/py310_t2_cu118/bin/ngp", line 33, in <module>
    sys.exit(load_entry_point('robot-vision', 'console_scripts', 'ngp')())
  File "/home/gao/opt/anaconda3/envs/py310_t2_cu118/bin/ngp", line 25, in importlib_load_entry_point
    return next(matches).load()
  File "/home/gao/opt/anaconda3/envs/py310_t2_cu118/lib/python3.10/importlib/metadata/__init__.py", line 171, in load
    module = import_module(match.group('module'))
  File "/home/gao/opt/anaconda3/envs/py310_t2_cu118/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/gao/projects/control/robot-vision/robot_vision/dataset/dex_nerf/__init__.py", line 1, in <module>
    from robot_vision.dataset.dex_nerf.run_all import main as run_dex_nerf
  File "/home/gao/projects/control/robot-vision/robot_vision/dataset/dex_nerf/run_all.py", line 12, in <module>
    from robot_vision.dataset.dex_nerf.train_nerf import train_ngp
  File "/home/gao/projects/control/robot-vision/robot_vision/dataset/dex_nerf/train_nerf.py", line 18, in <module>
    import pyngp as ngp
ImportError: /home/gao/opt/anaconda3/envs/py310_t2_cu118/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/gao/projects/vision/instant-ngp/build/pyngp.cpython-310-x86_64-linux-gnu.so)
```
this is due to gcc not up-to-date in anaconda, you have to update gcc to the latest one. e.g.
```shell
conda install -c conda-forge gcc=12.1.0
```