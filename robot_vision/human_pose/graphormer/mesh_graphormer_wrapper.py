import os
import gc
import sys
import copy
import click
import torch
import numpy as np
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
from typing import Union, List, Tuple, Dict
from marshmallow_dataclass import dataclass

from robot_vision.meta.install_dependencies import get_install_dir
sys.path.append(str(get_install_dir() / "opendr"))
sys.path.append(str(get_install_dir() / "opendr/contexts"))

from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.utils.geometric_layers import orthographic_projection

from src.utils.renderer import Renderer
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat

from robot_utils import console
from robot_utils.cv.io.io_cv import write_rgb
from robot_utils.cv.geom.projection import pinhole_projection_image_to_camera
from robot_utils.torch.torch_utils import init_torch
from robot_utils.serialize.schema_numpy import DictNumpyArray
from robot_utils.serialize.dataclass import load_dataclass, default_field
from robot_utils.py.filesystem import get_ordered_files, create_path, validate_path

from robot_vision.human_pose.graphormer.model_cfg import MG_configs
from robot_vision.human_pose.graphormer.mano_cfg import J_NAME
from robot_vision.human_pose.utils.occlusion import hand_auto_base_point_selection
from robot_vision.human_pose.mediapipe.utils.filter import draw_kalman_results

install_dir = get_install_dir()
os.chdir(str(install_dir / "graphormer"))

transform = transforms.Compose([
    # transforms.Resize(224, antialias=True),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, antialias=True),
    transforms.CenterCrop(224),
])


@dataclass
class MeshGraphormerConfig:
    seed: int = 88
    num_workers: int = 4
    use_gpu: bool = True
    input_feat_dim: List[int] = default_field([2051, 512, 128])
    hidden_feat_dim: List[int] = default_field([1024, 256, 64])
    which_gcn: List[int] = default_field([0, 0, 1])
    config_name: str = ''
    mesh_type: str = 'hand'
    arch: str = 'hrnet-w64'

    model_name_or_path: str = str(validate_path(
        install_dir / "graphormer/src/modeling/bert/bert-base-uncased", throw_error=True)[0])
    resume_checkpoint: str = MG_configs.get("graphormer_checkpoint", None)

    # c, h, w -> (2, 0, 1) -> w, c, h -> (1, 2, 0) -> c, h, w
    # h, w, c -> (2, 0, 1) -> c, h, w -> (1, 2, 0) -> h, w, c

    def __post_init__(self):
        os.environ['OMP_NUM_THREADS'] = str(self.num_workers)
        self.output_feat_dim = self.input_feat_dim[1:] + [3]


class MeshGraphormerWrapper:
    def __init__(self, cfg: Union[Path, dict, str, MeshGraphormerConfig] = None):
        console.rule("[bold cyan]loading Mesh Graphormer for hand pose estimation")
        self.c = load_dataclass(MeshGraphormerConfig, cfg)

        self.device = init_torch(self.c.seed, use_gpu=self.c.use_gpu)

        self.mano_model = MANO().to(self.device)
        self.mano_model.layer = self.mano_model.layer.cuda()
        self.mesh_sampler = Mesh(
            filename=str(install_dir / "graphormer/src/modeling/data/mano_downsampling.npz")
        )

        self.renderer = Renderer(faces=self.mano_model.face)

        trans_encoder = []

        for i in range(len(self.c.output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(
                self.c.config_name if self.c.config_name else self.c.model_name_or_path
            )

            config.output_attentions = False
            config.img_feature_dim = self.c.input_feat_dim[i]
            config.output_feature_dim = self.c.output_feat_dim[i]
            hidden_size = self.c.hidden_feat_dim[i]
            intermediate_size = int(hidden_size * 2)

            if self.c.which_gcn[i] == 1:
                config.graph_conv = True
                # logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = self.c.mesh_type
            config.num_hidden_layers = 4
            config.hidden_size = hidden_size
            config.num_attention_heads = 4
            config.intermediate_size = intermediate_size

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            console.log("Init model from scratch.")
            trans_encoder.append(model)

        # create backbone model
        if self.c.arch == 'hrnet':
            hrnet_yaml = MG_configs["hrnet_yaml"]
            hrnet_checkpoint = MG_configs["hrnet_checkpoint"]
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            console.log('=> loading hrnet-v2-w40 model')
        elif self.c.arch == 'hrnet-w64':
            hrnet_yaml = MG_configs["hrnet_yaml_w64"]
            hrnet_checkpoint = MG_configs["hrnet_checkpoint_w64"]
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            console.log('=> loading hrnet-v2-w64 model')
        else:
            console.log("=> using pre-trained model '{}'".format(self.c.arch))
            backbone = models.__dict__[self.c.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        console.log(f'Graphormer encoders total parameters: {total_params}')
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        console.log(f'Backbone total parameters: {backbone_total_params}')

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        self.graphormer_model = Graphormer_Network(None, config, backbone, trans_encoder)

        console.log(f"Loading state dict from checkpoint {self.c.resume_checkpoint}")
        # workaround approach to load sparse tensor in graph conv.
        state_dict = torch.load(self.c.resume_checkpoint)
        self.graphormer_model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        # update configs to enable attention outputs
        setattr(self.graphormer_model.trans_encoder[-1].config, 'output_attentions', True)
        setattr(self.graphormer_model.trans_encoder[-1].config, 'output_hidden_states', True)
        self.graphormer_model.trans_encoder[-1].bert.encoder.output_attentions = True
        self.graphormer_model.trans_encoder[-1].bert.encoder.output_hidden_states = True
        for iter_layer in range(4):
            self.graphormer_model.trans_encoder[-1].bert.encoder.layer[
                iter_layer].attention.self.output_attentions = True
        for inter_block in range(3):
            setattr(self.graphormer_model.trans_encoder[-1].config, 'device', self.device)

        self.graphormer_model.to(self.device)
        console.log(f"[bold cyan]Mesh Graphormer model is ready")

    def detect_hand_on_image_batch(
            self,
            image_array_raw: List[np.ndarray],          # (T, h, w, 3)
            depth_array_raw: List[np.ndarray],          # (T, h, w)
            cam_intrinsics: np.ndarray,                 # (3, 3)
            patch_bbox_array: DictNumpyArray,           # [hand_name, (T_i, 4)]
            sides: List[str],
            viz_path: Path,
            viz: bool = False,
            mask_array_raw: List[np.ndarray] = None,    # (T, h, w)
            time_idx: DictNumpyArray = None,            # [hand_name, (T_i, )]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Estimate the hand pose on a image batch
        Args:
            image_array_raw: (T, h, w, 3)
            depth_array_raw: (T, h, w)
            cam_intrinsics: (3, 3)
            patch_bbox_array: Dict[hand_name, (T_i, 4)]
            sides: (n_hands, )
            viz_path:
            viz:
            mask_array_raw: (T, h, w)
            time_idx: Dict[hand_name, (T_i, )]

        Returns: uv (n_hands, T, 21, 2) values in range u in [0, w] and v in [0, h]; xyz: (n_hands, T, 21, 3)

        """
        uv, xyz, base_point_idx = {}, {}, {}
        n_hands = len(patch_bbox_array.data.keys())
        n_frames = len(image_array_raw)
        viz_image_uv, viz_image_mesh = [None] * n_frames, [None] * n_frames

        with torch.no_grad():
            for hand_idx in range(n_hands):
                name = f"hand_{hand_idx:>02d}"
                patch_bbox = patch_bbox_array.data[name]      # (T_i, 4)
                side = sides[hand_idx]
                _uv, _xyz, _base_point_idx = [], [], []
                for idx, (image, depth, bbox) in enumerate(
                        zip(image_array_raw, depth_array_raw, patch_bbox)):
                    if idx not in time_idx.data[name]:
                        continue
                    # __uv: (21, 2), __xyz: (21, 3), _viz_image_xxx: (h, w, 3)
                    __uv, __xyz, _viz_image_uv, _viz_image_mesh = self.detect_hand_on_each_frame(
                        image, bbox, side, viz, viz_image_uv[idx], viz_image_mesh[idx])

                    if mask_array_raw is None:
                        visible_idx = np.arange(__uv.shape[0], dtype=int)
                    else:
                        mask = mask_array_raw[idx]
                        h, w = mask.shape[:2]
                        visible_idx = np.where(
                            (0 < __uv[:, 0]) & (__uv[:, 0] < w) &
                            (0 < __uv[:, 1]) & (__uv[:, 1] < h) &
                            np.logical_not(mask[__uv[:, 1], __uv[:, 0]])
                        )[0]

                    __base_point_idx = hand_auto_base_point_selection(__xyz, __uv, visible_idx)

                    base_point_xyz = pinhole_projection_image_to_camera(__uv[__base_point_idx], depth, cam_intrinsics)
                    __xyz = __xyz - __xyz[__base_point_idx] + base_point_xyz

                    viz_image_uv[idx] = _viz_image_uv
                    viz_image_mesh[idx] = _viz_image_mesh

                    _base_point_idx.append(__base_point_idx)
                    _uv.append(__uv)
                    _xyz.append(__xyz)
                base_point_idx[name] = np.array(_base_point_idx)
                uv[name] = np.array(_uv)
                xyz[name] = np.array(_xyz)

        if viz:
            p_uv = create_path(viz_path / f"uv")
            p_mesh = create_path(viz_path / f"mesh")
            for i in range(n_frames):
                if viz_image_uv[i] is None or viz_image_mesh[i] is None:
                    continue
                write_rgb(filename=str(p_uv/f"{i:>06d}.jpg"), img=viz_image_uv[i], bgr2rgb=True)
                write_rgb(filename=str(p_mesh/f"{i:>06d}.jpg"), img=viz_image_mesh[i], bgr2rgb=True)

        return uv, xyz

    def detect_hand_on_each_frame(
            self,
            image: np.ndarray,  # (h, w, 3)
            bbox: np.ndarray,   # (4, )
            handedness: str,
            viz: bool = False,
            viz_image_uv: np.ndarray = None,    # (h, w, 3)
            viz_image_mesh: np.ndarray = None,  # (h, w, 3)
    ):
        """

        Args:
            image: (h, w, 3), the raw RGB image
            bbox: the (u1, v1, u2, v2) bounding box of the cropped image
            handedness: either left or right
            viz:
            viz_image_uv: only provide this image, if you need to plot directly on the image
            viz_image_mesh: only provide this image, if you need to plot directly on the image

        Returns:

        """
        cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # (224, 224, 3)
        if handedness == "left":
            cropped_image = np.ascontiguousarray(np.flip(cropped_image, axis=1))

        img_tensor = transform(cropped_image)
        batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()

        # forward-pass:
        # - pred_camera: (b, 3) or (3, )        - pred_3d_joints: (b, 21, 3)    - hidden_states: Tuple
        # - pred_vertices_sub: (b, 195, 3)      - pred_vertices: (b, 778, 3)    - att: Tuple
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = self.graphormer_model(
            batch_imgs, self.mano_model, self.mesh_sampler)

        # obtain 3d joints (landmarks) from full mesh
        pred_3d_joints_from_mesh = self.mano_model.get_3d_joints(pred_vertices)  # (b, 21, 3)
        ref_joint_idx = J_NAME.index('Wrist')
        # zero the wrist position
        pred_3d_pelvis = pred_3d_joints_from_mesh[:, ref_joint_idx, :]  # (b, 3)
        pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]  # (b, 21, 3)
        pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]  # (b, 778, 3)
        pred_vertices = pred_vertices.squeeze().detach().cpu().numpy()

        xyz = pred_3d_joints_from_mesh.squeeze().cpu().detach().numpy()  # (b, 21, 3) or (21, 3)
        # local frame: x to the right, z front, (0, 0, 0) is the origin of the camera frame (cropped image)
        if handedness == "left":
            for joint in xyz:
                joint[0] = -joint[0]

        pred_2d_joints_from_mesh = orthographic_projection(
            pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous()
        )
        joint_2d = pred_2d_joints_from_mesh.squeeze().cpu().detach().numpy()  # (b, 21, 2) or (21, 2)

        flip = -1 if handedness == "left" else 1
        uv = np.zeros_like(joint_2d, dtype=int)
        uv[:, 0] = (joint_2d[:, 0] * flip * 112 + 112).astype(int) + bbox[0]
        uv[:, 1] = (joint_2d[:, 1] * 112 + 112).astype(int) + bbox[1]

        pred_camera = pred_camera.detach().cpu().numpy()  # (b, 3) or (3, )

        if viz:
            # visualize the uv coordinates on RGB images using MediaPipes utility function
            if viz_image_uv is None:
                viz_image_uv = copy.deepcopy(image)
            h, w = viz_image_uv.shape[:2]
            viz_image_uv = draw_kalman_results(viz_image_uv, uv / np.array([w, h]))

            # visualize the 3D mesh on RGB images using the render tools from MeshGraphormer
            if viz_image_mesh is None:
                viz_image_mesh = copy.deepcopy(image)  # (h, w, 3)
            cropped_image = viz_image_mesh[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # (224, 224, 3)
            cropped_image = transform_visualize(cropped_image).numpy().transpose(1, 2, 0)  # value in range [0, 1]
            cropped_image = cropped_image[:, ::-1] if handedness == "left" else cropped_image

            focal_length = 1000
            res = cropped_image.shape[1]
            camera_t = np.array([
                pred_camera[1], pred_camera[2], 2 * focal_length / (res * pred_camera[0] + 1e-9)
            ])  # (3, )
            rendered_image = self.renderer.render(
                pred_vertices, camera_t=camera_t, img=cropped_image,
                use_bg=True, focal_length=focal_length, body_color="light_blue"
            )  # (224, 224, 3), in value [0, 1]
            rendered_image = rendered_image[:, ::-1] if handedness == "left" else rendered_image
            viz_image_mesh[bbox[1]:bbox[3], bbox[0]:bbox[2]] = rendered_image * 255
        torch.cuda.empty_cache()
        return uv, xyz, viz_image_uv, viz_image_mesh

    @staticmethod
    def get_joint_names() -> List[str]:
        return J_NAME


def get_3d_joints(image_path: Path, viz: bool = False, ref_joint_idx: int = 0, side: str = "right"):
    from robot_utils.cv.io.io_cv import load_rgb
    console.log("Run inference")

    image_files = get_ordered_files(image_path, to_str=True, pattern=['.png'])
    image_array = np.array([load_rgb(f, bgr2rgb=True) for f in image_files])  # (T, h, w, c)
    ic(image_array.shape)
    output_dir = create_path(image_path / "hand_pred_output")
    mg_model = MeshGraphormerWrapper()
    traj_3d_this_hand, wrist_uv_this_hand = mg_model.run(image_array, output_dir, viz, side=side)
    return traj_3d_this_hand, wrist_uv_this_hand


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--root_path", "-p", type=str, help="the absolute path to the training root folder of demo")
@click.option("--hand", "-h", type=str, help="the absolute path to the training root folder of demo")
def main(root_path, hand):
    import robot_utils.log as log
    p = Path("/home/gao/dataset/kvil/test/demo_pour_new_structure/test/hand_00")

    log.disable_logging()
    get_3d_joints(p, viz=True, side="left")
    log.enable_logging()


if __name__ == "__main__":
    main()

# Note: python robot_vision/human_pose/graphormer/get_joints_from_mesh.py
