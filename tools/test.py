"""
模型测试与评估入口脚本。

加载五折交叉验证保存的权重，在测试集上计算 IoU、Dice、Precision 等指标，
并输出 CSV 汇总与最佳模型可视化。

用法示例::

    python tools/test.py
    python tools/test.py --config configs/config.yaml
"""

import os
import sys
import argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs import load_config
from data_pipeline import TestDataset
from models import UNet, UNetPlusPlus, HMTUNet
from utils import (
    compute_iou,
    compute_dice,
    compute_precision,
    compute_recall,
    compute_specificity,
    compute_accuracy,
)
from utils.visualization import visualize_random_samples


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    在测试集上评估模型，返回各项指标均值。
    """
    model.eval()
    metrics_accum = {
        "iou": [],
        "dice": [],
        "prec": [],
        "rec": [],
        "acc": [],
        "spec": [],
    }

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for pred, target in zip(preds, masks_np):
                pred = np.squeeze(pred)
                target = np.squeeze(target)
                # 若掩码值域为 0-255，则归一化到 0-1
                if target.max() > 1.0:
                    target = target / 255.0

                metrics_accum["iou"].append(compute_iou(pred, target, threshold=threshold))
                metrics_accum["dice"].append(compute_dice(pred, target, threshold=threshold))
                metrics_accum["prec"].append(
                    compute_precision(pred, target, threshold=threshold)
                )
                metrics_accum["rec"].append(compute_recall(pred, target, threshold=threshold))
                metrics_accum["acc"].append(compute_accuracy(pred, target, threshold=threshold))
                metrics_accum["spec"].append(
                    compute_specificity(pred, target, threshold=threshold)
                )

    mean_metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
    return mean_metrics


def main(cfg: Dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    hp = cfg["hyper_params"]
    checkpoint_cfg = cfg["checkpoint"]
    output_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size: int = hp["img_size"]
    threshold: float = hp["threshold"]
    batch_size: int = hp.get("test_batch_size", 1)

    # 测试数据路径
    test_image_dir = data_cfg.get(
        "test_images", os.path.join(data_cfg["processed_root"], "test", "images")
    )
    test_mask_dir = data_cfg.get(
        "test_masks", os.path.join(data_cfg["processed_root"], "test", "masks")
    )

    # 权重路径列表
    model_paths: List[str] = checkpoint_cfg.get("test_model_paths", [])
    if not model_paths:
        raise ValueError("配置文件中 'checkpoint.test_model_paths' 为空，请填写待测试模型路径。")

    # 默认使用 HMTUNet；如需切换，可在配置中增加字段或在命令行扩展
    ModelClass = HMTUNet

    test_dataset = TestDataset(
        test_image_dir,
        test_mask_dir,
        img_size=(img_size, img_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=hp.get("num_workers", 4),
    )

    records = []
    for path in model_paths:
        if not os.path.isfile(path):
            print(f"[跳过] 权重文件不存在: {path}")
            continue

        print(f"\n>> Evaluating model: {path}")
        model = ModelClass(
            n_channels=hp.get("in_channels", 3),
            n_classes=hp.get("num_classes", 1),
        ).to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except Exception as e:
            print(f"[错误] 加载权重失败 {path}: {e}")
            continue

        metrics = evaluate(model, test_loader, device, threshold=threshold)
        metrics["model"] = os.path.basename(path)
        records.append(metrics)
        print("Result:", metrics)

    if not records:
        print("未成功评估任何模型，请检查权重路径与数据路径。")
        return

    df = pd.DataFrame(records)
    df = df[["model", "iou", "dice", "prec", "rec", "acc", "spec"]]

    print("\n=== All folds results ===")
    print(df.to_string(index=False))

    best = df.loc[df["dice"].idxmax()]
    print(f"\n*** Best model: {best['model']} (Dice={best['dice']:.4f}) ***")

    output_root = output_cfg.get("root", "./outputs")
    _ensure_dir(output_root)
    csv_path = os.path.join(output_root, output_cfg.get("csv_name", "evaluation_summary.csv"))
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")

    # 最佳模型可视化
    best_path = model_paths[df["dice"].idxmax()]
    print(f"\nVisualizing samples from best model: {best['model']}")
    best_model = ModelClass(
        n_channels=hp.get("in_channels", 3),
        n_classes=hp.get("num_classes", 1),
    ).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    visualize_random_samples(
        best_model, test_dataset, device, num_samples=4, threshold=threshold, img_size=(img_size, img_size)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2017 皮肤病变分割测试脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
