import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Union, List, Literal

from pathlib import Path
from marshmallow_dataclass import dataclass

from robot_utils import console
from robot_utils.cv.io.io_cv import load_rgb, load_depth, load_mask
from robot_utils.cv.image.op_torch import get_imagenet_normalization
from robot_utils.cv.correspondence.finder_torch import create_non_correspondences, pixel_correspondences
from robot_utils.py.utils import load_dict_from_yaml, load_dataclass_from_yaml
from robot_vision.dataset.dex_nerf.utils import get_intrinsics_from_transform
from robot_vision.utils.correspondence.correspondence_augmentation import random_domain_randomize_background, \
    random_image_and_indices_mutation, apply_randomization_full_image
from robot_utils.torch.torch_utils import get_device, get_image_mesh_grid
from robot_vision.utils.utils import get_default_dataset_path


@dataclass
class DCNDataConfig:
    data_root: str
    obj: str
    all: list
    train: list
    eval: list = None
    mode: str = "train"
    namespace: str = ""
    rgb_ext: str = ".jpg"
    dep_ext: str = ".png"
    mask_ext: str = ".png"
    scale: float = 1.0
    width: int = 1280
    height: int = 720
    image_mean: list = None
    image_std: list = None
    n_images_total: int = 0
    n_images: dict = None
    random_image_b_mode: Literal["reverse", "forward", "forward_moving", "fixed_window", "fixed_dist"] = "fixed_window"
    enable_flip_augmentation: bool = False
    domain_randomize: bool = True
    n_samples: int = 100
    sample_matches_only_off_mask: bool = True
    use_image_b_mask: bool = True
    num_non_matches_per_match_on_mask: int = 150
    num_non_matches_per_match_off_mask: int = 150

    random_color_channel_permute: bool = True
    random_channel_permute_prob: float = 0.4
    domain_randomize_prob: float = 0.4
    bgr2rgb: bool = True

    def get_scene_list(self, mode: str = ""):
        if not mode:
            mode = self.mode
        if mode == "train":
            return self.train
        elif mode == "eval":
            return self.eval
        elif mode == "all":
            return self.all
        else:
            raise NotImplementedError(f"{mode} not supported")

    def __post_init__(self):
        if self.mode != "train":
            self.domain_randomize = False

    def set_mode(self, mode: str = 'train'):
        self.mode = mode
        if self.mode != "train":
            self.domain_randomize = False


class PDCDataset(Dataset):
    def __init__(
            self,
            config: Union[str, Path] = "config.yaml",
            direct_on_gpu: bool = False,
            mode: str = None
    ):
        """
        Args:
            config: absolute path to 'data_config.yaml'
            direct_on_gpu: True, the data will be directly transferred to GPU before loaded into dataloader
            mode: 'train' or 'eval' mode
        """
        super(PDCDataset, self).__init__()
        console.rule("[bold blue] Initializing Dense Correspondence Dataset")
        console.log(f"reading config from {config}")
        self.c = load_dataclass_from_yaml(DCNDataConfig, config)
        self.device = get_device(direct_on_gpu)
        if mode is not None:
            self.c.set_mode(mode)
        self._init_scene()
        console.rule("[bold blue]done")

    def _init_scene(self):
        self.data_root = get_default_dataset_path() / "kvil/objects" / self.c.obj
        if not self.data_root.exists():
            console.log(f"[bold red]DCN dataset: the working directory {self.data_root} does not exist")
            exit(1)
        if not self.c.namespace:
            console.log(f"[red]namespace invalid")
            exit(1)
        console.log(f"[bold blue]working on {self.data_root} in namespace {self.c.namespace} with object {self.c.obj}")

        # scene
        self.scene_list_all = [self.data_root / scene for scene in self.c.get_scene_list("all")]
        self.scene_list = [self.data_root / scene for scene in self.c.get_scene_list()]
        if len(self.scene_list) == 0:
            console.print(f"[bold red]There are no scenes for mode {self.c.mode} in this dataset")
            exit(1)

        # images
        self._n_images = {}
        for scene in self.scene_list:
            self._n_images[scene.stem] = self.c.n_images[scene.stem]
        self.num_images_total = sum(self._n_images.values())
        console.log(f"number of images in all scenes {self._n_images}")
        if self.num_images_total == 0:
            console.rule("[bold red] no images found in your dataset")
            exit(1)

        self.img_mean, self.img_std_dev = torch.tensor(self.c.image_mean), torch.tensor(self.c.image_std)
        self._load_all_pose_data()

        self.w = int(self.c.scale * self.c.width)
        self.h = int(self.c.scale * self.c.height)
        self.image_wh = (self.w, self.h)
        scale = (self.w / self.c.width, self.h / self.c.height)

        # read transforms from any one of the scenes
        transform_file = self.scene_list[0] / "transforms.json"
        transform = load_dict_from_yaml(transform_file)
        self.intrinsics = get_intrinsics_from_transform(transform, scale)

        # norm_transform = transforms.Normalize(self.img_mean, self.img_std_dev)
        norm_transform = get_imagenet_normalization()
        self.rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])

        self.progress = 0.

    def __len__(self):
        return self.num_images_total

    def set_progress(self, progress: float):
        self.progress = progress

    def _get_image_b_index(self, image_a_idx: int, scene_name: Path):
        n_images_in_scene = self._n_images[scene_name.stem]
        if self.c.random_image_b_mode == "reverse":
            # ----- (1) reversed progress ----------------------------------
            # loss decreases slowly but in the end converge to relatively low loss value, results is much better
            progress = 1 - self.progress
            delta = random.randint(
                1,
                max(10, int(progress * 0.4 * n_images_in_scene))
            )
        elif self.c.random_image_b_mode == "forward":
            # ----- (2) forward progress: always start from 1 ---------------
            # Not good, start to diverge after some progress
            delta = random.randint(
                # max(1, int(self.progress * 0.1 * n_images_in_scene)),
                1,
                max(10, int(self.progress * 0.4 * n_images_in_scene))
            )
        elif self.c.random_image_b_mode == "forward_moving":
            # ----- (3) forward progress: with moving window -----------------
            # Not good, start to diverge after some progress
            delta = random.randint(
                max(1, int(self.progress * 0.1 * n_images_in_scene)),
                max(10, int(self.progress * 0.4 * n_images_in_scene))
            )
        elif self.c.random_image_b_mode == "fixed_window":
            # ----- (4) fixed window between 1 and 10 ------------------------
            # works nicely if the two images are not significantly different, very smooth correspondence detection
            delta = random.randint(1, 15)
        elif self.c.random_image_b_mode == "fixed_dist":
            # ----- (5) fixed offset with predefined ratio --------------------
            # not good enough, sometimes not converging to low loss, results then degenerates
            delta = random.randint(1, int(n_images_in_scene * 0.3))

        image_b_idx = int((image_a_idx + delta) % n_images_in_scene)
        return image_b_idx

    def __getitem__(self, index):
        scene_name = self.get_random_scene(n_scenes=1)[0]  # type: Path
        image_a_idx = self.get_random_image_index(scene_name)

        # Step: 1) read images
        # image a
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, image_a_idx)

        # image b
        image_b_idx = self._get_image_b_index(image_a_idx, scene_name)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, image_b_idx)

        lower, upper, rotation = self.get_aabb(scene_name)

        # Step: 2) find correspondence matches
        if self.c.sample_matches_only_off_mask:
            correspondence_mask = image_a_mask
        else:
            correspondence_mask = None

        # Step: 2.1) compute correspondence
        uv_a, uv_b = pixel_correspondences(
            image_a_depth, image_a_pose, image_b_depth, image_b_pose, intrinsic=self.intrinsics,
            img_a_mask=correspondence_mask, img_b_mask=image_b_mask, n_samples=self.c.n_samples, device=self.device,
            lower=lower, upper=upper, rotation=rotation,
        )

        if uv_a is None:
            console.log(f"[red]no matches found, returning {image_a_idx} and {image_b_idx} from scene {scene_name}")
            return torch.tensor([])

        # Step: 2.2) data augmentation
        if self.c.domain_randomize and np.random.random() < self.c.domain_randomize_prob:
            # Note: the domain randomization should be implemented in torch
            # image_a_rgb = random_domain_randomize_background(image_a_rgb, image_a_mask)
            # image_b_rgb = random_domain_randomize_background(image_b_rgb, image_b_mask)
            image_a_rgb = apply_randomization_full_image(image_a_rgb)
            image_b_rgb = apply_randomization_full_image(image_b_rgb)

        # Note: the domain randomization should be implemented in torch, and use variables to specify the image size
        if self.c.enable_flip_augmentation:
            [image_a_rgb, image_a_mask], uv_a = random_image_and_indices_mutation([image_a_rgb, image_a_mask], uv_a)
            [image_b_rgb, image_b_mask], uv_b = random_image_and_indices_mutation([image_b_rgb, image_b_mask], uv_b)

        if self.c.random_color_channel_permute and np.random.random() < self.c.random_channel_permute_prob:
            permute = np.random.permutation(3)
            image_a_rgb = np.array(image_a_rgb)[..., permute]
            image_b_rgb = np.array(image_b_rgb)[..., permute]

        # # Step: 3) find non_correspondences
        # image_b_mask_torch = torch.from_numpy(image_b_mask).to(self.device)
        #
        # image_b_shape = image_b_depth.shape
        # image_width = image_b_shape[1]
        # image_height = image_b_shape[0]
        #
        # uv_b_masked_non_matches = create_non_correspondences(
        #     uv_b, image_b_shape, num_non_matches_per_match=self.c.num_non_matches_per_match_on_mask,
        #     img_b_mask=image_b_mask, within_mask=True
        # )
        # uv_b_background_non_matches = create_non_correspondences(
        #     uv_b, image_b_shape, num_non_matches_per_match=self.c.num_non_matches_per_match_off_mask,
        #     img_b_mask=image_b_mask if self.c.use_image_b_mask else None,
        #     within_mask=False
        # )

        # Step: 4) preparing the returns
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)

        return image_a_rgb, image_b_rgb, uv_a, uv_b

    def get_random_scene(self, n_scenes: int = 1) -> List[Path]:
        """
        return a random scene (absolute path) corresponding to self.c.mode
        """
        return random.sample(self.scene_list, k=n_scenes)

    def get_random_image_index(self, scene: Path) -> int:
        """
        Returns a random image index from a given scene
        """
        return np.random.randint(self._n_images[scene.stem])

    def _load_all_pose_data(self):
        """
        Efficiently pre-loads all pose data for the scenes. This is because when used as part of torch DataLoader in threaded way it behaves strangely.
        """
        self._pose_data = {}
        for scene_path in self.scene_list:
            scene_name = scene_path.stem
            if scene_name not in self._pose_data:
                console.log(f"[green]Loading pose data for scene {scene_name}")
                pose_data_filename = scene_path / self.c.namespace / 'pose_data.yaml'
                self._pose_data[scene_name] = load_dict_from_yaml(pose_data_filename)["poses"]

    def get_pose_from_scene_name_and_idx(self, scene_name: Path, img_idx: int) -> np.ndarray:
        """
        Args:
            scene_name: absolute Path object to scene
            img_idx:

        Returns: pose transformation matrix (4, 4)
        """
        return np.array(self._pose_data[scene_name.stem][img_idx])

    def get_uv_all(self, device: str = None):
        if device is None:
            device = self.device
        return get_image_mesh_grid(self.h, self.w).to(device)

    def get_rgbd_mask_pose(self, scene_name: Path, img_idx: int):
        """
        Returns: PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, a 4x4 numpy array
        """
        img_path = scene_name / self.c.namespace
        # rgb: (h, w, c)  (720, 1280, 3)     depth: (h, w, 1)  (720, 1280, 1)     mask: (h, w)     (720, 1280)
        rgb = cv2.resize(load_rgb(img_path / f"shade/{img_idx:>04d}{self.c.rgb_ext}", bgr2rgb=self.c.bgr2rgb), self.image_wh)
        depth = cv2.resize(load_depth(img_path / f"depth/{img_idx:>04d}{self.c.dep_ext}"), self.image_wh)
        mask = cv2.resize(load_mask(img_path / f"mask/{img_idx:>04d}{self.c.mask_ext}"), self.image_wh)
        pose = self.get_pose_from_scene_name_and_idx(scene_name, img_idx)

        return rgb, depth, mask, pose

    def get_aabb(self, scene_name: Path):
        aabb_file = scene_name / self.c.namespace / "aabb.yaml"
        aabb_dict = load_dict_from_yaml(aabb_file)

        lower = torch.tensor(aabb_dict["lower"], device=self.device).float()
        upper = torch.tensor(aabb_dict["upper"], device=self.device).float()
        rotation = torch.tensor(aabb_dict["rotation"], device=self.device).float()
        return lower, upper, rotation

    def get_rgb(self, scene_name: Path, img_idx: int):
        img_path = scene_name / self.c.namespace
        # (h, w, c)  (720, 1280, 3) * scale
        return cv2.resize(load_rgb(img_path / f"shade/{img_idx:>04d}{self.c.rgb_ext}", bgr2rgb=self.c.bgr2rgb), self.image_wh)

    def get_random_rgb_filename(self):
        scene_name = self.get_random_scene(1)[0]
        img_idx = self.get_random_image_index(scene_name)
        return scene_name / self.c.namespace / f"shade/{img_idx:>04d}{self.c.rgb_ext}"

    def get_rgb_from_path(self, path: Path):
        return cv2.resize(load_rgb(path, bgr2rgb=self.c.bgr2rgb), self.image_wh)

    def get_depth_from_path(self, path: Path):
        return cv2.resize(load_depth(path), self.image_wh)
