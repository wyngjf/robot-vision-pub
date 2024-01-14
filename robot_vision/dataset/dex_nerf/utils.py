import numpy as np
from typing import Union, Tuple, List


def get_intrinsics_from_transform(
        transform: dict,
        scale: Union[float, Tuple[float, float], np.ndarray, List[float]] = 1.0
) -> np.ndarray:
    if isinstance(scale, float):
        scale = np.ones(2) * scale
    return np.array([
        [scale[0] * transform["fl_x"], 0., scale[0] * transform["cx"]],
        [0, scale[1] * transform["fl_y"], scale[1] * transform["cy"]],
        [0, 0, 1]
    ])


