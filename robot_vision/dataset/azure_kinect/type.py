import numpy as np
from marshmallow_dataclass import dataclass
from robot_utils.serialize.dataclass import default_field
from robot_utils.serialize.schema_numpy import NumpyArray


@dataclass
class Calibration:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


@dataclass
class AzureKinectCameraParam:
    rgb: Calibration = Calibration()
    image_size: list = default_field([])
    intrinsic_matrix: NumpyArray = None

    def get_intrinsics(self, scale: float = 1.0, **kwargs):
        if self.intrinsic_matrix is not None:
            return self.intrinsic_matrix.reshape((3, 3)).T.tolist()

        if isinstance(scale, float):
            scale = [scale, scale]
        return [
            [self.rgb.fx * scale[0], 0,  self.rgb.cx * scale[0]],
            [0, self.rgb.fy * scale[1],  self.rgb.cy * scale[1]],
            [0, 0, 1]
        ]
