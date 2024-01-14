from marshmallow_dataclass import dataclass


@dataclass
class ZEDCalibration:
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
class ZEDCameraParam:
    left: ZEDCalibration = None
    right: ZEDCalibration = None
    image_size: list = None
    rotation: list = None
    translation: list = None

    def get_intrinsics(self, flag_left: bool, scale: float = 1.0, **kwargs):
        if isinstance(scale, float):
            scale = [scale, scale]
        view = self.left if flag_left else self.right
        return [
            [view.fx * scale[0], 0,  view.cx * scale[0]],
            [0, view.fy * scale[1],  view.cy * scale[1]],
            [0, 0, 1]
        ]
