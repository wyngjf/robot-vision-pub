from setuptools import setup, find_packages
setup(
    name='robot-vision',
    version='1.0.0',
    author='Jianfeng Gao',
    author_email='jianfeng.gao@kit.edu',
    description="This package contains computer vision models for robotic development in Python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "trimesh",
        "transforms3d",
        "pyrender",
        "rich",
        "dearpygui",
        "tensorflow",
        "tensorboard_logger",
        "open3d",
        "polyscope",
        "pathos",
        "filterpy",

    ],
    extras_require={
        "mediapipe": ["mediapipe"]
    },
    entry_points={
        "console_scripts": [
            "robot_vision_install=robot_vision.meta:install",
            "robot_vision_dep_on=robot_vision.meta:generate",
            "dcn_train=robot_vision.dcn.train:main",
            "dcn_generate=robot_vision.dcn.generate_dcn_train:generate_dcn_train_cfg",
            "dcn_viz=robot_vision.dcn.viz.dcn_heatmap:main",
            "dcn_viz_heatmap=robot_vision.dcn.viz.dcn_masked_heatmap:main",
            "zed_export_all=robot_vision.dataset.stereo.zed.export_all:main",
            "ngp=robot_vision.dataset.dex_nerf.run_all:main",
        ]
    }
)
