from marshmallow_dataclass import dataclass
from typing import Dict, List
from robot_utils.serialize.schema_numpy import NumpyArray


@dataclass
class Classes:
    supercategory:              str
    id:                         int
    name:                       str
    keypoints:                  List[str]
    skeleton:                   List[List[int]]


@dataclass
class MMWholeBodyPoseMeta:
    dataset_name:               str
    num_keypoints:              int
    keypoint_id2name:           Dict[str, str]
    keypoint_name2id:           Dict[str, int]
    upper_body_ids:             List[int]
    lower_body_ids:             List[int]
    flip_indices:               List[int]
    flip_pairs:                 List[List[int]]
    keypoint_colors:            NumpyArray
    num_skeleton_links:         int
    skeleton_links:             List[List[int]]
    skeleton_link_colors:       NumpyArray
    dataset_keypoint_weights:   NumpyArray
    sigmas:                     NumpyArray
    # CLASSES:                    Classes

    def get_left_hand_nodes(self):
        return [self.keypoint_id2name[str(i)] for i in self.get_hand_landmark_id_l()]

    def get_right_hand_nodes(self):
        return [self.keypoint_id2name[str(i)] for i in self.get_hand_landmark_id_r()]

    @staticmethod
    def get_hand_landmark_id_l():
        return list(range(91, 112))

    @staticmethod
    def get_hand_landmark_id_r():
        return list(range(112, 133))

    def get_left_wrist_id(self):
        return self.keypoint_name2id["left_wrist"]

    @staticmethod
    def get_right_wrist_id():
        return self.keypoint_name2id["right_wrist"]




