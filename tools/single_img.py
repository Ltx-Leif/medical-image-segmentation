"""
单张图像推理与轮廓可视化脚本。

加载指定模型权重，对单张图像进行分割预测，并将真实掩码与预测掩码轮廓叠加在原图上显示。

用法示例::

    python tools/single_img.py
    python tools/single_img.py --config configs/config.yaml --model HMTUNet
    python tools/single_img.py --image path/to/img.jpg --mask path/to/mask.png --weight path/to.pth
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs import load_config
from models import UNet, UNetPlusPlus, HMTUNet
from utils.visualization import visualize_single_contour


_MODEL_REGISTRY = {
    "UNet": UNet,
    "UNetPlusPlus": UNetPlusPlus,
    "HMTUNet": HMTUNet,
}


def load_model(
    model_name: str,
    weight_path: str,
    device: torch.device,
    n_channels: int = 3,
    n_classes: int = 1,
) -> torch.nn.Module:
    """
    根据模型名称加载权重并返回 eval 模式下的模型实例。

    Raises:
        ValueError: 模型名称不在注册表中。
        FileNotFoundError: 权重文件不存在。
        RuntimeError: 权重加载失败。
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"未知模型 '{model_name}'。可用选项: {list(_MODEL_REGISTRY.keys())}"
        )
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"权重文件未找到: {weight_path}")

    ModelClass = _MODEL_REGISTRY[model_name]
    model = ModelClass(n_channels=n_channels, n_classes=n_classes).to(device)
    try:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"加载权重失败 ({weight_path}): {e}")
    model.eval()
    return model


def main(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    hp = cfg["hyper_params"]
    checkpoint_cfg = cfg["checkpoint"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size: int = hp["img_size"]
    threshold: float = hp["threshold"]

    # 优先级：命令行参数 > 配置文件
    model_name = args.model or "HMTUNet"
    weight_path = args.weight or checkpoint_cfg.get("single_model_path", "")
    image_path = args.image or checkpoint_cfg.get("single_image_path", "")
    mask_path = args.mask or checkpoint_cfg.get("single_mask_path", "")

    if not weight_path or not image_path or not mask_path:
        raise ValueError(
            "必须提供模型权重、图像路径和掩码路径（通过命令行或配置文件）。"
        )

    # 校验文件存在性
    for p, desc in [(image_path, "图像"), (mask_path, "掩码"), (weight_path, "权重")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{desc}文件未找到: {p}")

    model = load_model(
        model_name,
        weight_path,
        device,
        n_channels=hp.get("in_channels", 3),
        n_classes=hp.get("num_classes", 1),
    )

    # 图像预处理
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img_t = transform(img).to(device).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_bin = (prob > threshold).astype(np.uint8)

    gt_bin = (np.array(mask.resize((img_size, img_size))) > 0).astype(np.uint8)

    # 原图 resize 到显示尺寸
    img_display = np.array(img.resize((img_size, img_size))) / 255.0

    visualize_single_contour(
        image=img_display,
        gt_mask=gt_bin,
        pred_mask=pred_bin,
        figure_title=f"{model_name} | {os.path.basename(image_path)}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单张图像分割可视化")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型名称 (UNet / UNetPlusPlus / HMTUNet)")
    parser.add_argument("--weight", type=str, default=None, help="权重文件路径")
    parser.add_argument("--image", type=str, default=None, help="测试图像路径")
    parser.add_argument("--mask", type=str, default=None, help="真实掩码路径")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config, args)
