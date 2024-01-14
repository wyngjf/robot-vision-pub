import torch
from typing import Union, Dict, Tuple
from marshmallow_dataclass import dataclass
import robot_utils.torch.resnet as resnet_dilated
from robot_utils.py.utils import load_dataclass
from robot_utils.torch.base_model import BaseModel, ModelConfig


@dataclass
class DCNConfig(ModelConfig):
    descriptor_dimension: int = 3
    image_width: int = 0
    image_height: int = 0
    image_mean: list = None
    image_std: list = None

    model_name: str = ""


class DenseCorrespondenceNetwork(BaseModel):
    def __init__(self, config: Union[str, Dict, None] = None, model_path: str = None):
        super(DenseCorrespondenceNetwork, self).__init__(config, model_path)

    def _load_config(self, config: Dict) -> None:
        self.c = load_dataclass(DCNConfig, config)

    def _build_model(self) -> None:
        if "Resnet" in self.c.model_name:
            self.model = getattr(resnet_dilated, self.c.model_name)(num_classes=self.c.descriptor_dimension)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x)

    def forward_descriptor_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (3, h, w) or (batch, 3, h, w)

        Returns: (b, h * w, D)
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # transform to shape (1,3,h,w)
        return self.forward(image.to(self.device))\
            .view(image.shape[0], self.c.descriptor_dimension, self.c.image_width * self.c.image_height)\
            .permute(0, 2, 1)
    
    def forward_single_image_tensor(self, image: torch.Tensor, resize_hw: Tuple[int, int] = None) -> torch.Tensor:
        """
        Args:
            image: (3, h, w) or (batch=1, 3, h, w)
            resize: to resize the results

        Returns: descriptor image (h, w, D)
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # transform to shape (1,3,h,w)
        descriptor_image = self.forward(image.to(self.device))
        if resize_hw is not None:
            descriptor_image = torch.nn.functional.interpolate(
                descriptor_image, size=resize_hw, mode='bilinear', align_corners=True
            )
        # shape (1,D,h,w) -> (D,h,w) -> (h,w,D)
        return descriptor_image.squeeze(0).permute(1, 2, 0)
