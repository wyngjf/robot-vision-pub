import numpy as np
from typing import List
from sklearn.neighbors import LocalOutlierFactor

from robot_utils.cv.opencv import erode_mask


def crop_obj_pcl(
        pcl: np.ndarray,
        mask: np.ndarray,
        filter_mask: np.ndarray = None,
        remove_outlier: bool = True,
        erosion_radius: int = -1,
) -> np.ndarray:
    """
    crop the object from the point cloud
    Args:
        pcl: (n_points, 3)
        mask: (h, w) where h * w = n_points, the mask of the object
        filter_mask: (n_points, )
        remove_outlier: whether to automatically remove outliers
        erosion_radius: if you need to erode the mask, set to values larger than 0

    Returns: the cropped point cloud of the object

    """
    if erosion_radius > 0:
        mask = erode_mask(mask, erosion_radius)

    obj_mask = mask.flatten() if filter_mask is None else np.logical_and(mask.flatten(), filter_mask)
    idx = np.where(obj_mask)[0]
    if len(idx) < 1:
        return None

    pcl = pcl[idx]
    if len(idx) < 30 or not remove_outlier:
        return pcl

    clf = LocalOutlierFactor(n_neighbors=30, contamination='auto')
    in_out_lier_indicator = clf.fit_predict(pcl)
    inlier_idx = np.where(in_out_lier_indicator == 1)[0]
    return pcl[inlier_idx]


def example_of_crop():
    # read depth image and convert to point cloud
    pcl: np.ndarray = np.random.random(size=(720*1920, 3))

    pcl_dist_to_cam_center = np.linalg.norm(pcl, axis=-1)
    filter_mask = np.logical_and(pcl_dist_to_cam_center > 0.02, pcl_dist_to_cam_center < 3.0)

    # read mask image
    from robot_utils.cv.io.io_cv import load_mask
    obj_mask: np.ndarray = load_mask(filename="", as_binary=True)

    # crop pcl for the obj
    obj_pcl = crop_obj_pcl(pcl, obj_mask, filter_mask, erosion_radius=5, remove_outlier=True)

    # setup polyscope plot with a color


def get_indices_from_pred_dict(pred_dict, obj_name: str) -> List[int]:
    return [i for i in range(len(pred_dict["labels"])) if obj_name == pred_dict["labels"][i]]


def get_indices_of_item_in_list(source_list, item: str) -> List[int]:
    return [i for i in range(len(source_list)) if item == source_list[i]]
