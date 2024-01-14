import torch
import numpy as np
from pathlib import Path
from typing import Union
from marshmallow_dataclass import dataclass

from robot_utils.cv.io.io_cv import load_rgb
from robot_utils.py.utils import load_dataclass
from robot_utils.torch.torch_utils import get_device
from robot_vision.utils.utils import get_default_checkpoints_path

from raft_core.raft import RAFT, RAFTConfig
from raft_core.utils import flow_viz
from raft_core.utils.utils import InputPadder


@dataclass
class RAFTWrapperConfig:
    """
    The RAFT model wrapper config
    Args:
        checkpoint: the abs path to the checkpoint file (.pth)
        flag_use_gpu: use torch.device gpu

    """
    checkpoint:         str = None
    flag_use_gpu:       bool = True
    iterations:         int = 6

    model_cfg:          RAFTConfig = RAFTConfig()


def to_tensor(image: np.ndarray, device):
    return torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)


class RAFTWrapper:
    def __init__(self, cfg: Union[Path, dict, str, RAFTWrapperConfig] = None):
        self.c = load_dataclass(RAFTWrapperConfig, cfg)
        self.device = get_device(self.c.flag_use_gpu)

        self._load_model()

    def _load_model(self):
        if self.c.checkpoint is None:
            checkpoint_path = str(get_default_checkpoints_path() / "raft/models/raft-things.pth")
        else:
            checkpoint_path = self.c.checkpoint
        checkpoint = torch.load(checkpoint_path)

        model = torch.nn.DataParallel(RAFT(self.c.model_cfg))
        model.load_state_dict(checkpoint)
        self.model = model.module
        self.model.to(self.device)
        self.model.eval()

    def run(self, image1: torch.Tensor, image2: torch.Tensor):
        """
        Args:
            image1:
            image2:
        Returns:
        """
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = self.model(
            image1.to(self.device), image2.to(self.device), iters=20, test_mode=True
        )
        flo = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()
        flow_img = flow_viz.flow_to_image(flo)
        del image1, image2, flow_low, flow_up
        return np.round(flo).astype(int), flow_img

    def get_depth_in_mm(
            self,
            image1: torch.Tensor,
            image2: torch.Tensor,
            base_line_mm: float,
            focal_length: float,
            max_depth_in_mm: float = 5000.0
    ):
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = self.model(
            image1.to(self.device), image2.to(self.device), iters=20, test_mode=True
        )
        flow = flow_up[0]
        epsilon = 1e-6
        disparity = (base_line_mm * focal_length) / (torch.linalg.norm(flow, dim=0) + epsilon)

        depth_map = torch.divide(torch.tensor([base_line_mm * focal_length]).to(disparity), disparity + 1e-10)
        depth_map[depth_map > max_depth_in_mm] = max_depth_in_mm
        return depth_map

    def estimate_depth(self, img_l, img_r, camera_base_line, focal_length_w, depth_unit_in_meter=True):
        depth = self.get_depth_in_mm(
            to_tensor(img_l, self.device), to_tensor(img_r, self.device), camera_base_line, focal_length_w
        )
        return depth.detach().cpu().numpy()



def demo():
    from robot_utils.py.filesystem import get_ordered_files, create_path
    from pathlib import Path
    from robot_utils.cv.io.io_cv import write_rgb

    path = Path("/home/gao/dataset/checkpoints/raft/test_data/masked_rgb")
    filenames = get_ordered_files(path, pattern=[".png"])

    model = RAFTWrapper()
    images = [
        torch.from_numpy(load_rgb(image_file, bgr2rgb=True)).permute(2, 0, 1).float()
        for image_file in filenames
    ]

    p = create_path(path.parent / "output_flow")

    with torch.no_grad():
        for i in range(len(images) - 1):
            flow, flow_img = model.run(images[i].unsqueeze(0), images[i+1].unsqueeze(0))
            write_rgb(p / f"{i:>06d}.png", flow_img, bgr2rgb=True)


def demo_depth():
    from robot_utils.py.filesystem import get_ordered_files, create_path
    from pathlib import Path
    from robot_utils.cv.io.io_cv import write_rgb, write_colorized_depth
    from torchvision import transforms
    from robot_vision.utils.utils import get_root_path
    device = "cuda:0"

    def to_tensor(image: np.ndarray):
        return torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    p = Path("/home/gao/dataset/kvil/demo/spoon_to_plate_to_tablemat/to_center/recordings/20230816_171251/images")
    img1 = load_rgb(p / "left_000132.jpg")
    img2 = load_rgb(p / "right_000132.jpg")
    model = RAFTWrapper()
    depth = model.get_depth_in_mm(to_tensor(img1), to_tensor(img2), base_line_mm=120.0, focal_length=710.98)
    depth = depth.detach().cpu().numpy().astype(np.uint16)
    path = create_path(get_root_path() / "output/raft") / "depth.png"
    write_colorized_depth(path, depth)

    cam_intrinsics = np.array([
        [710.9813842773438, 0, 635.4081420898438], [0, 710.9813842773438, 354.359619140625], [0, 0, 1]
    ])
    from robot_utils.cv.geom.projection import pinhole_projection_image_to_camera, get_wh
    wh = get_wh(depth.shape[1], depth.shape[0])
    pcl = pinhole_projection_image_to_camera(wh, depth, cam_intrinsics)

    import polyscope as ps
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)
    ps.init()
    ps.set_length_scale(1.)
    ps.set_up_dir("neg_y_up")
    ps.set_navigation_style("free")
    ps.set_ground_plane_mode("none")
    ps.register_point_cloud(
        f"pcl", pcl, radius=0.01, enabled=True, point_render_mode="sphere"
    )
    ps.show()


if __name__ == '__main__':
    demo()
