"""
评估指标与损失函数。

提供二元图像分割任务常用的 IoU、Dice、Precision、Recall、Specificity、Accuracy
以及训练用的 DiceLoss 与 dice_coefficient 函数。
"""

from typing import Union
import numpy as np
import torch
import torch.nn as nn


def _get_tp_fp_fn_tn(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> tuple:
    """
    计算二元分割结果的 TP、FP、FN、TN。

    Args:
        pred: 预测概率图（numpy 数组或 torch tensor）。
        target: 真实二值图（numpy 数组或 torch tensor）。
        threshold: 二值化阈值。

    Returns:
        (tp, fp, fn, tn) 元组，类型与 ``pred`` 一致（若为 numpy 则转为 float32 tensor）。
    """
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)

    if not torch.is_tensor(target):
        target = torch.as_tensor(target, dtype=torch.bool, device=pred.device)
    else:
        target = target.bool().to(pred.device)

    pred_binary = (pred > threshold).bool()

    tp = (pred_binary & target).sum()
    fp = (pred_binary & ~target).sum()
    fn = (~pred_binary & target).sum()
    tn = (~pred_binary & ~target).sum()

    return tp, fp, fn, tn


def compute_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    计算 Jaccard Index (IoU)。

    公式: ``TP / (TP + FP + FN)``
    """
    tp, fp, fn, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    return float(iou.item())


def compute_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    计算 Dice 系数 (F1-Score)。

    公式: ``2 * TP / (2 * TP + FP + FN)``
    """
    tp, fp, fn, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    return float(dice.item())


def compute_precision(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    计算精确率 (Precision)。

    公式: ``TP / (TP + FP)``
    """
    tp, fp, _, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    return float(precision.item())


def compute_recall(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    计算召回率 (Recall / Sensitivity)。

    公式: ``TP / (TP + FN)``
    """
    tp, _, fn, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    return float(recall.item())


def compute_specificity(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    计算特异度 (Specificity)。

    公式: ``TN / (TN + FP)``
    """
    _, fp, _, tn = _get_tp_fp_fn_tn(pred, target, threshold)
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    return float(specificity.item())


def compute_accuracy(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    计算准确率 (Accuracy)。

    公式: ``(TP + TN) / (TP + TN + FP + FN)``
    """
    tp, fp, fn, tn = _get_tp_fp_fn_tn(pred, target, threshold)
    accuracy = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)
    return float(accuracy.item())


class DiceLoss(nn.Module):
    """
    Dice 损失函数，适用于二分类分割任务。

    输入应为模型输出的 **logits**（未经 sigmoid），
    内部会自动应用 sigmoid 后再计算 Dice。

    Args:
        epsilon: 平滑项，防止除零，默认 ``1e-6``。
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            raise ValueError(
                f"DiceLoss 期望输入与目标形状一致，但获得 {inputs.shape} vs {targets.shape}"
            )
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.epsilon) / (
            probs.sum() + targets.sum() + self.epsilon
        )
        return 1.0 - dice


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算 batch 级别的 Dice 系数（用于验证阶段）。

    Args:
        pred: 模型输出 logits，形状 ``(B, 1, H, W)`` 或 ``(B, H, W)``。
        target: 真实掩码，形状与 ``pred`` 一致。
        epsilon: 平滑项。

    Returns:
        标量 Dice 值（torch.Tensor）。
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + epsilon) / (
        pred_flat.sum() + target_flat.sum() + epsilon
    )
    return dice
