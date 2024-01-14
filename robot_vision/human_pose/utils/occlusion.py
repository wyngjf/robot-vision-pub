import numpy as np


def filter_points_with_mask(uv: np.ndarray, mask: np.ndarray):
    """

    Args:
        uv: (n_points, 2)
        mask: (h, w) the binary mask -> note: make sure the size of mask matches with the 2d position (u,v) of joints

    Returns: idx_list_points_not_occluded: list[int]

    """
    h, w = mask.shape
    return np.where((0 < uv[:, 0] < w) & (0 < uv[:, 1] < h) & np.logical_not(mask[uv[:, 1], uv[:, 0]]))[0]


def hand_auto_base_point_selection(
        xyz: np.ndarray,
        uv: np.ndarray,
        visible_idx: np.ndarray
) -> int:
    """
    Args:
        xyz: the 3d position (x, y, z) of joints, the closer to the camera, the smaller z is
        uv: the 2d position (u, v) of joints
        visible_idx: the indices of landmarks that are not occluded by any object

    Returns: the index that is most likely the base point

    """
    # mediapipe hand landmark index:

    # thumb_idx_list = [1, 2, 3, 4]
    # index_finger_idx_list = [5, 6, 7, 8]
    # middle_finger_idx_list = [9, 10, 11, 12]
    # ring_finger_idx_list = [13, 14, 15, 16]
    # pinky_idx_list = [17, 18, 19, 20]
    # wrist_idx = [0]

    idx_proirity_list = [9, 5, 13, 17, 10, 6, 14, 18, 0, 11, 7, 15, 19, 12, 8, 16, 20, 1, 2, 3, 4]
    finger_idx_dict = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
    }
    palm_idx_list = [0, 1, 5, 9, 13, 17]
    finger_name_list = finger_idx_dict.keys()
    # ic(finger_name_list)

    # remove joints at the edge of the hands to improve the perfoemance
    hand_com = np.mean(uv, axis=0)
    idx_list_edge_joints = []
    for finger in finger_name_list:
        finger_idx_list = finger_idx_dict[finger]
        joint_com_dist_list = []
        for idx in finger_idx_list:
            joint = uv[idx]
            joint_com_dist_list.append(np.linalg.norm(joint - hand_com))
        idx_list_edge_joints.append(finger_idx_list[np.argmax(np.array(joint_com_dist_list))])

    # detect self occlusion
    self_occluded_idx_list = []
    for finger in finger_name_list:
        if "thumb" in finger:
            continue
        finger_idx_list = finger_idx_dict[finger]
        palm_joint_idx_list = []
        non_palm_joint_idx_list = []
        for joint_idx in finger_idx_list:
            if joint_idx in palm_idx_list:
                palm_joint_idx_list.append(joint_idx)
            else:
                non_palm_joint_idx_list.append(joint_idx)
        u_values = []
        v_values = []
        z_values = []
        for joint_idx in non_palm_joint_idx_list:
            joint_2d = uv[joint_idx]
            joint_3d = xyz[joint_idx]
            u_values.append(joint_2d[0])
            v_values.append(joint_2d[1])
            z_values.append(joint_3d[2])
        u_values = np.array(u_values)
        v_values = np.array(v_values)
        z_values = np.array(z_values)
        for joint_idx in palm_joint_idx_list:
            joint_2d = uv[joint_idx]
            joint_3d = xyz[joint_idx]
            if (np.min(u_values) < joint_2d[0] < np.max(u_values)) and (
                    np.min(v_values) < joint_2d[1] < np.max(v_values)) and (joint_3d[2] > np.mean(z_values)):
                self_occluded_idx_list.append(joint_idx)

    for joint_idx in idx_proirity_list:
        if (joint_idx not in idx_list_edge_joints) and (joint_idx in visible_idx) and (
                joint_idx not in self_occluded_idx_list):
            return joint_idx

    for joint_idx in idx_proirity_list:
        if (joint_idx in visible_idx) and (joint_idx not in self_occluded_idx_list):
            return joint_idx

    return 9
