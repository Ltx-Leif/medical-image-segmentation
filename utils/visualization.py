"""
可视化辅助函数。

提供分割结果对比图、轮廓叠加图、雷达图等常用可视化工具。
"""

from typing import Tuple, Optional
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import measure


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将图像张量转换为可用于 ``plt.imshow`` 的 numpy 数组。

    Args:
        tensor: 形状为 ``(C, H, W)`` 的 torch 张量，值域通常 ``[0, 1]``。

    Returns:
        形状为 ``(H, W, C)`` 的 numpy 数组。
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img


def visualize_random_samples(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_samples: int = 5,
    threshold: float = 0.5,
    img_size: Tuple[int, int] = (256, 256),
) -> None:
    """
    随机抽取若干样本，绘制原图、真实掩码与预测掩码对比。

    Args:
        model: 已加载权重的分割模型（处于 eval 模式）。
        dataset: 数据集对象，需支持 ``dataset[idx] -> (image, mask)``。
        device: 计算设备。
        num_samples: 可视化样本数。
        threshold: 预测概率二值化阈值。
        img_size: 期望的图像显示尺寸，用于掩码 resize。
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, 0)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            x = image.unsqueeze(0).to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)
            pred_mask = (prob[0, 0].cpu().numpy() > threshold).astype(np.uint8)

            gt_mask = mask.squeeze(0).cpu().numpy()
            # 若掩码值域为 0-255，则归一化到 0-1
            if gt_mask.max() > 1.0:
                gt_mask = gt_mask / 255.0

            ax_orig, ax_gt, ax_pred = axes[row]
            ax_orig.imshow(tensor_to_image(image))
            ax_orig.set_title("Original Image")
            ax_orig.axis("off")

            ax_gt.imshow(gt_mask, cmap="gray")
            ax_gt.set_title("Ground Truth")
            ax_gt.axis("off")

            ax_pred.imshow(pred_mask, cmap="gray")
            ax_pred.set_title("Predicted Mask")
            ax_pred.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_single_contour(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    figure_title: str = "Single Model, Single Sample Visualization",
    legend_labels: Optional[Tuple[str, str]] = None,
) -> None:
    """
    在单张图像上绘制真实掩码轮廓（绿色）与预测掩码轮廓（红色）。

    Args:
        image: 原始图像，numpy 数组，值域 ``[0, 1]`` 或 ``[0, 255]``，形状 ``(H, W, 3)``。
        gt_mask: 真实二值掩码，形状 ``(H, W)``。
        pred_mask: 预测二值掩码，形状 ``(H, W)``。
        figure_title: 图表标题。
        legend_labels: 图例文字，默认 ("Ground Truth (Green)", "Prediction (Red)")。
    """
    if legend_labels is None:
        legend_labels = ("Ground Truth (Green)", "Prediction (Red)")

    contours_gt = measure.find_contours(gt_mask, 0.5)
    contours_pred = measure.find_contours(pred_mask, 0.5)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(image, 0, 1))
    for c in contours_gt:
        plt.plot(c[:, 1], c[:, 0], linewidth=2, color="g")
    for c in contours_pred:
        plt.plot(c[:, 1], c[:, 0], linewidth=2, color="r")

    handles = [
        plt.Line2D([0], [0], color="g", lw=2),
        plt.Line2D([0], [0], color="r", lw=2),
    ]
    plt.legend(handles, legend_labels, loc="upper right")
    plt.axis("off")
    plt.title(figure_title)
    plt.tight_layout()
    plt.show()
