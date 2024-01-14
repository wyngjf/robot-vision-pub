from robot_vision.utils.utils import get_default_checkpoints_path
from robot_utils.py.filesystem import validate_file

training_root = get_default_checkpoints_path()

MG_configs = dict(
    graphormer_checkpoint=str(validate_file(
        training_root / "MeshGraphormer/graphormer_hand_state_dict.bin", throw_error=True)[0]),
    hrnet_yaml="",
    hrnet_checkpoint="",
    hrnet_yaml_w64=str(validate_file(
        training_root / "hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml", throw_error=True)[0]),
    hrnet_checkpoint_w64=str(validate_file(
        training_root / "hrnet/hrnetv2_w64_imagenet_pretrained.pth", throw_error=True)[0]),
)
