import mmcv
import numpy as np
import json_tricks as json
from pathlib import Path
from marshmallow_dataclass import dataclass
from typing import Literal, Union, Sequence, List, Dict, Any
from scipy.optimize import linear_sum_assignment

from mmengine.structures import InstanceData
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector

from robot_utils import console
from robot_utils.cv.io.io_cv import load_rgb
from robot_utils.py.utils import load_dataclass
from robot_utils.py.interact import user_warning
from robot_utils.torch.torch_utils import get_device
from robot_utils.py.filesystem import create_path, validate_file
from robot_utils.math.similarity.sim_np import cosine_similarity_matching_rows, euc_l2_similarity_matching_row
from robot_utils.serialize.dataclass import save_to_yaml, dump_data_to_yaml, load_dataclass
from robot_vision.utils.utils import get_default_checkpoints_path, get_root_path
from robot_vision.meta.install_dependencies import get_install_dir
from robot_vision.human_pose.mmpose.rtmpose.meta_cfg import MMWholeBodyPoseMeta


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def split_instances(instances: InstanceData) -> List[Dict[str, Any]]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),
            if 'bbox_scores' in instances:
                result['bbox_score'] = float(instances.bbox_scores[i])
        results.append(result)

    return results


@dataclass
class RTMPoseConfig:
    det_config:         str = None      # Config file for human bbox detection
    det_checkpoint:     str = None      # Checkpoint file for detection model
    pose_config:        str = None      # Config file for pose estimation
    pose_checkpoint:    str = None      # Checkpoint file for pose estimation model

    flag_use_gpu:       bool = True

    det_cat_id:         int = 0         # Category id for bounding box detection model
    bbox_thr:           float = 0.3     # Bounding box score threshold
    nms_thr:            float = 0.3     # IoU threshold for bounding box NMS
    kpt_thr:            float = 0.6     # Visualizing keypoint thresholds # TODO rename

    show_viz_live:      bool = False
    draw_bbox:          bool = False
    draw_heatmap:       bool = False    # Draw heatmap predicted by the model
    show_kpt_idx:       bool = True     # Whether to show the index of keypoints
    skeleton_style:     Literal['mmpose', 'openpose'] = "mmpose"
    radius:             int = 4         # Keypoint radius for visualization
    thickness:          int = 2         # Link thickness for visualization
    alpha:              float = 0.8     # The transparency of bboxes  # TODO rename

    def __post_init__(self):
        if self.det_config is None:
            det_cfg_path = "mmpose/projects/rtmpose/rtmdet/person"
            self.det_config = str(get_install_dir() / det_cfg_path / "rtmdet_m_640-8xb32_coco-person.py")

        if self.det_checkpoint is None:
            filename = "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
            self.det_checkpoint = str(get_default_checkpoints_path() / "mmpose/rtmdet" / filename)

        if self.pose_config is None:
            pose_cfg_path = "mmpose/projects/rtmpose/rtmpose/wholebody_2d_keypoint"
            self.pose_config = str(get_install_dir() / pose_cfg_path / "rtmpose-x_8xb32-270e_coco-wholebody-384x288.py")

        if self.pose_checkpoint is None:
            filename = "rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth"
            self.pose_checkpoint = str(get_default_checkpoints_path() / "mmpose/rtmpose" / filename)

        self.device = get_device(self.flag_use_gpu)


class RTMPoseWrapper:
    def __init__(self, cfg: Union[Path, dict, str, RTMPoseConfig] = None):
        user_warning(f"This class {self.__class__.__name__} is not complete (see TODOs), use at your own risk")
        self.c = load_dataclass(RTMPoseConfig, cfg)

        self._load_model()
        self.pred_instances_list = []

    def _load_model(self):
        # build detector
        self.detector = init_detector(
            self.c.det_config, self.c.det_checkpoint, device=self.c.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(
            self.c.pose_config,
            self.c.pose_checkpoint,
            device=self.c.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=self.c.draw_heatmap))))

        # build visualizer
        self.pose_estimator.cfg.visualizer.radius = self.c.radius
        self.pose_estimator.cfg.visualizer.alpha = self.c.alpha
        self.pose_estimator.cfg.visualizer.line_width = self.c.thickness
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        self.visualizer.set_dataset_meta(
            self.pose_estimator.dataset_meta, skeleton_style=self.c.skeleton_style
        )

        self.meta_info = self.get_meta_from_file()
        self.node_idx = dict(
            left_hand=self.meta_info.get_hand_landmark_id_l(),
            right_hand=self.meta_info.get_hand_landmark_id_r()
        )

    def process_one_image(self, img: ImagesType):
        """Visualize predicted keypoints (and heatmaps) of one image."""

        # predict bbox
        det_result = inference_detector(self.detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == self.c.det_cat_id,
                                       pred_instance.scores > self.c.bbox_thr)]
        bboxes = bboxes[nms(bboxes, self.c.nms_thr), :4]

        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)

        # show the results
        if isinstance(img, str):
            img = mmcv.imread(img, channel_order='rgb')
        elif isinstance(img, np.ndarray):
            img = mmcv.bgr2rgb(img)

        if self.visualizer is not None:
            self.visualizer.add_datasample(
                'result',
                img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=self.c.draw_heatmap,
                draw_bbox=self.c.draw_bbox,
                show_kpt_idx=self.c.show_kpt_idx,
                skeleton_style=self.c.skeleton_style,
                show=self.c.show_viz_live,
                wait_time=0,
                kpt_thr=self.c.kpt_thr)

        # if there is no instance detected, return None
        return data_samples.get('pred_instances', None)

    def reset(self):
        self.pred_instances_list = []

    def detect_on_images(self, image: ImagesType, viz: bool = True):
        pred_instances = self.process_one_image(image)
        instance_list = split_instances(pred_instances)
        self.pred_instances_list = self._assign_human_id(self.pred_instances_list, instance_list)

        if viz and pred_instances is not None:
            img_vis = self.visualizer.get_image()
            return self.pred_instances_list, mmcv.rgb2bgr(img_vis)

        return self.pred_instances_list, None

    def _assign_human_id(self, instance_list_src, instance_list_tgt):
        # New human will always be appended to the end of the list, if some human is not detected in the middle,
        # the corresponding entry should yield None for instance_list_target. So that the human ids match between frames
        # The number of elements should only increase over time (including None elements)
        # TODO add Kalman Filter for the case that prev estimation has more human instances than the current estimation
        if len(self.pred_instances_list) == 0:
            return instance_list_tgt

        kpt_uvs_list_src = [np.array(instance["keypoints"]) for instance in instance_list_src]  # (n, 113, 2)
        kpt_uvs_list_tgt = [np.array(instance["keypoints"]) for instance in instance_list_tgt]  # (m, 113, 2)
        kpt_score_list_src = [np.array(instance["keypoint_scores"]) for instance in instance_list_src]
        kpt_score_list_tgt = [np.array(instance["keypoint_scores"]) for instance in instance_list_tgt]

        n_src, n_tgt = len(kpt_uvs_list_src), len(kpt_uvs_list_tgt)
        similarity_matrix = np.zeros((n_src, n_tgt))
        for i in range(n_src):
            for j in range(n_tgt):
                # similarity = cosine_similarity_matching_rows(
                #     kpt_uvs_list_src[i], kpt_uvs_list_tgt[j], kpt_score_list_src[i], kpt_score_list_tgt[j])
                # )
                similarity = euc_l2_similarity_matching_row(
                    kpt_uvs_list_src[i], kpt_uvs_list_tgt[j], kpt_score_list_src[i], kpt_score_list_tgt[j]
                )
                similarity_matrix[i, j] = similarity

        # row_indices, col_indices = linear_sum_assignment(similarity_matrix, True)
        row_indices, col_indices = linear_sum_assignment(similarity_matrix)
        if n_src < n_tgt:
            # Note: append the new human instances at the end of the list
            rearrange_order = col_indices.tolist().extend([i for i in range(n_tgt) if i not in col_indices])
        else:
            # TODO here we try to ignore the human instances if they are not detected anymore in this step,
            #  It's fine for K-VIL demos, but this has to be fixed in the future for multi-human collaboration tasks
            rearrange_order = col_indices
        # console.log(f"sim: \n {similarity_matrix}, \n "
        #             f"r: {row_indices}, c: {col_indices}, order {rearrange_order}, n_sr: {n_src}, n_tgt: {n_tgt}")
        return [instance_list_tgt[i] for i in rearrange_order]

    def check_hand_ids(self, human_instance):
        landmark_array = np.array(human_instance["keypoints"], dtype=int)
        hand_array_l = landmark_array[self.node_idx["left_hand"]]
        hand_array_r = landmark_array[self.node_idx["right_hand"]]
        human_instance["left_hand"] = hand_array_l
        human_instance["right_hand"] = hand_array_r

        score_array = np.array(human_instance["keypoint_scores"])
        hand_scores_l = score_array[self.node_idx["left_hand"]]
        hand_scores_r = score_array[self.node_idx["right_hand"]]
        if np.median(hand_scores_l) < self.c.kpt_thr:
            human_instance["left_hand"] = None
        if np.median(hand_scores_r) < self.c.kpt_thr:
            human_instance["right_hand"] = None

        hand_landmark_dist = np.linalg.norm(hand_array_l - hand_array_r, axis=-1).mean()
        # TODO without Kalman Filter, the handedness tends to flip around quite often, as future work
        if hand_landmark_dist < 10:
            if np.median(hand_scores_l) > np.median(hand_scores_r):
                human_instance["right_hand"] = None
                # console.log(f"left {np.median(hand_scores_l)} > right {np.median(hand_scores_r)}, choose left")
            else:
                human_instance["left_hand"] = None
                # console.log(f"right {np.median(hand_scores_r)} > left {np.median(hand_scores_l)}, choose right")

        return human_instance

    def get_meta_info(self, out_filename: Union[Path, str] = None):
        meta = self.pose_estimator.dataset_meta
        if out_filename is not None:
            with open(out_filename, 'w') as f:
                json.dump(meta, f, indent='    ')
            console.log(f'[cyan]Meta info have been saved at {out_filename}')

        return meta

    @staticmethod
    def get_meta_from_file(meta_filename: Path = None):
        if meta_filename is None:
            meta_filename = get_root_path() / "human_pose/mmpose/rtmpose/meta.json"
        validate_file(meta_filename, throw_error=True)
        return load_dataclass(MMWholeBodyPoseMeta, meta_filename)

