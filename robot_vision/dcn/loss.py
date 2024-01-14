import torch
from torch.autograd import Variable
from marshmallow_dataclass import dataclass

from robot_utils import console
from robot_utils.torch.torch_utils import rand_choice_gpu, get_image_mesh_grid
from robot_utils.cv.image.op_torch import flatten_uv_tensor
from robot_utils.py.utils import load_dataclass_from_dict

from rich import inspect


def zero_loss():
    return Variable(torch.FloatTensor([0]).cuda())


def is_zero_loss(loss):
    return loss is None or loss.item() < 1e-20


@dataclass
class HeatmapLossConfig:
    use_decreasing_parameter: bool = False
    eta: float = 1.0  # heatmap loss configs
    sigma: float = 10.0
    use_decreasing_sample: bool = False
    decreasing_sample_num: int = 60
    width: int = 0
    height: int = 0
    w_heatmap: float = 0.8  # weights for heatmap loss, 1-w_heatmap for spatial expectation loss


class HeatmapLoss:
    def __init__(self, config: dict, device: str):
        self.device = device
        self.c = load_dataclass_from_dict(HeatmapLossConfig, config)

        self.sigma = torch.tensor([self.c.sigma]).to(self.device)
        self.eta = torch.tensor([self.c.eta]).to(self.device)

    def get(
            self,
            image_a_pred, image_b_pred,
            matches_a, matches_b,
            current_iter,
            uv_flatten
    ):
        """

        Args:
            image_a_pred: (1, 640*480, 3)
            image_b_pred: (1, 640*480, 3)
            matches_a: (1, n_match, 2)
            matches_b: (1, n_match, 2)
            current_iter:
            uv_flatten: (h*w, 2)

        Returns:

        """
        current_iter = torch.tensor(current_iter).float().to(self.device)

        # decreasing parameters
        if self.c.use_decreasing_parameter:
            sigma = 60 * torch.exp(-0.001 * current_iter) + self.sigma
            eta = 6 * torch.exp(-0.001 * current_iter) + self.eta
        else:
            sigma = self.sigma
            eta = self.eta

        if self.c.use_decreasing_sample:
            num_matches = self.c.decreasing_sample_num + (150 * torch.exp(-0.0005 * current_iter)).to(torch.int)
        else:
            num_matches = self.c.decreasing_sample_num

        num_matches = min(num_matches, matches_b.shape[0])  # n = num_matches
        idx = rand_choice_gpu(matches_b.shape[0], num_matches, shuffle=False)  # (n, )
        matches_a, matches_b = matches_a[idx], matches_b[idx]

        if num_matches < 1:
            print("num_matches is 0!")
            return zero_loss()

        uv_matches_b = matches_b.float()  # (n, 2)
        # desired heatmap # (1, h*w, 2) - (n, 1, 2) -> (n, h*w, 2) -> (n, h*w)
        heatmap_flat_true = torch.exp(-torch.div(
            (uv_flatten.unsqueeze(0) - uv_matches_b.unsqueeze(1)).norm(dim=-1).square(),
            sigma.square()
        ))

        # predicted heatmap (1, hw, d) -> (n, 1, d) - (1, hw, d) -> (n, hw, d) -> (n, hw)
        uv_a_flatten = flatten_uv_tensor(matches_a, self.c.width)
        heatmap_flat_est = torch.exp(-torch.div(
            (image_a_pred.squeeze()[uv_a_flatten].unsqueeze(1) - image_b_pred).norm(dim=-1).square(),
            eta.pow(2)
        ))

        # heatmap loss
        num_pixel = self.c.width * self.c.height
        loss_heatmap = 1.0/num_pixel * (heatmap_flat_est - heatmap_flat_true).pow(2).sum(dim=1)     # (100, )
        loss_heatmap = torch.div(loss_heatmap.sum(), loss_heatmap.shape[0])

        # H_tilde: the probability density function based on heatmap
        heatmap_pdf = torch.div(heatmap_flat_est, heatmap_flat_est.sum(dim=1, keepdim=True) + 1e-10)        # (n, w*h)

        # spatial expectation loss
        # computed the expected pixel locations, they should match with uv_matches_b
        # (1, h*w, 2)  (n, h*w, 1) -> (n, h*w, 2) -> (n, 2)
        expected_uv = (uv_flatten.unsqueeze(0) * heatmap_pdf.unsqueeze(2)).sum(dim=1)               # (n, 2)
        loss_spatial_pixel = torch.norm((uv_matches_b - expected_uv), p=1, dim=1)                   # (n, )
        loss_spatial_pixel = torch.div(loss_spatial_pixel.sum(), loss_spatial_pixel.shape[0])

        # total loss
        return self.c.w_heatmap * loss_heatmap + (1 - self.c.w_heatmap) * loss_spatial_pixel

    def get_vit_loss(
            self,
            image_a_pred, image_b_pred,
            matches_a, matches_b,
            current_iter,
            # uv_flatten
    ):
        """

        Args:
            image_a_pred: (b, c, h, w)
            image_b_pred: (b, c, h, w)
            matches_a: (1, n_match, 2)
            matches_b: (1, n_match, 2)
            current_iter:
            uv_flatten: (h*w, 2)

        Returns:

        """
        current_iter = torch.tensor(current_iter).float().to(self.device)

        # decreasing parameters
        if self.c.use_decreasing_parameter:
            sigma = 60 * torch.exp(-0.001 * current_iter) + self.sigma
            eta = 6 * torch.exp(-0.001 * current_iter) + self.eta
        else:
            sigma = self.sigma
            eta = self.eta

        if self.c.use_decreasing_sample:
            num_matches = self.c.decreasing_sample_num + (150 * torch.exp(-0.0005 * current_iter)).to(torch.int)
        else:
            num_matches = self.c.decreasing_sample_num

        num_matches = min(num_matches, matches_b.shape[0])  # n = num_matches
        idx = rand_choice_gpu(matches_b.shape[0], num_matches, shuffle=False)  # (n, )
        matches_a, matches_b = matches_a[idx], matches_b[idx]

        if num_matches < 1:
            print("num_matches is 0!")
            return zero_loss()

        b, c, h, w = image_b_pred.shape
        scale = image_b_pred.new_tensor([h / self.c.height, w / self.c.width])
        uv_matches_b = matches_b * scale  # (n, 2)
        uv_flatten = get_image_mesh_grid(h, w).to(image_b_pred)
        # desired heatmap # (1, h*w, 2) - (n, 1, 2) -> (n, h*w, 2) -> (n, h*w)
        heatmap_flat_true = torch.exp(-torch.div(
            (uv_flatten.unsqueeze(0) - uv_matches_b.unsqueeze(1)).norm(dim=-1).square(),
            sigma.square()
        ))

        # predicted heatmap (1, hw, d) -> (n, 1, d) - (1, hw, d) -> (n, hw, d) -> (n, hw)
        uv_a_flatten = flatten_uv_tensor((matches_a * scale).long(), w)
        image_a_pred = image_a_pred.squeeze().view(c, h*w).transpose(-2, -1)
        image_b_pred = image_b_pred.view(b, c, h*w).transpose(-2, -1)
        heatmap_flat_est = torch.exp(-torch.div(
            (image_a_pred[uv_a_flatten].unsqueeze(1) - image_b_pred).norm(dim=-1).square(),
            eta.pow(2)
        ))
        # console.rule("--")
        # console.log(heatmap_flat_true.shape)
        # console.log(heatmap_flat_est.shape)
        # heatmap loss
        num_pixel = h * w
        loss_heatmap = 1.0/num_pixel * (heatmap_flat_est - heatmap_flat_true).pow(2).sum(dim=1)     # (100, )
        # console.log(loss_heatmap.shape)
        loss_heatmap = torch.div(loss_heatmap.sum(), loss_heatmap.shape[0])
        # console.log(loss_heatmap)

        # H_tilde: the probability density function based on heatmap
        # console.log(f"sum {heatmap_flat_est.sum(dim=1, keepdim=True)}")
        heatmap_pdf = torch.div(heatmap_flat_est, heatmap_flat_est.sum(dim=1, keepdim=True) + 1e-10)        # (n, w*h)
        # console.log(heatmap_pdf)

        # spatial expectation loss
        # computed the expected pixel locations, they should match with uv_matches_b
        # (1, h*w, 2)  (n, h*w, 1) -> (n, h*w, 2) -> (n, 2)
        expected_uv = (uv_flatten.unsqueeze(0) * heatmap_pdf.unsqueeze(2)).sum(dim=1)               # (n, 2)
        loss_spatial_pixel = torch.norm((uv_matches_b - expected_uv), p=1, dim=1)                   # (n, )
        # console.log(loss_spatial_pixel.shape)
        loss_spatial_pixel = torch.div(loss_spatial_pixel.sum(), loss_spatial_pixel.shape[0])

        # total loss
        # console.log(loss_heatmap)
        # console.log(loss_spatial_pixel)
        # console.rule(" cc ")
        # exit()
        return self.c.w_heatmap * loss_heatmap + (1 - self.c.w_heatmap) * loss_spatial_pixel

