"""
模型训练入口脚本。

支持基于 YAML 配置的 5 折交叉验证训练，兼容 UNet、UNet++ 与 HMTUNet。

用法示例::

    python tools/train.py
    python tools/train.py --config configs/config.yaml
"""

import os
import sys
import glob
import copy
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# 将项目根目录加入 PYTHONPATH，以支持 models/、utils/ 等包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs import load_config
from data_pipeline import ISICDataset, get_transforms
from models import UNet, UNetPlusPlus, HMTUNet
from utils import DiceLoss, dice_coefficient


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_info: str,
) -> float:
    """执行一个训练 epoch，返回平均损失。"""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Training {epoch_info}", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, list):  # UNet++ deep supervision
            loss = sum(criterion(o, masks) for o in outputs) / len(outputs)
        else:
            loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch_info: str,
) -> float:
    """执行验证，返回平均 Dice 系数。"""
    model.eval()
    total_dice = 0.0
    pbar = tqdm(loader, desc=f"Validation {epoch_info}", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]
            dice = dice_coefficient(outputs, masks)
            total_dice += dice.item()
    return total_dice / len(loader)


def main(cfg: Dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    hp = cfg["hyper_params"]
    checkpoint_cfg = cfg["checkpoint"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size: int = hp["img_size"]
    batch_size: int = hp["batch_size"]
    epochs: int = hp["epochs"]
    k_folds: int = hp["k_folds"]
    lr: float = hp["lr"]
    weight_decay: float = hp["weight_decay"]
    patience: int = hp["early_stopping_patience"]
    model_save_root = checkpoint_cfg["root"]

    _ensure_dir(model_save_root)

    # 搜集训练图像与掩码路径
    train_img_dir = data_cfg.get("train_images", os.path.join(data_cfg["processed_root"], "train", "images"))
    train_mask_dir = data_cfg.get("train_masks", os.path.join(data_cfg["processed_root"], "train", "masks"))

    all_train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")))
    all_train_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.png")))

    if not all_train_images:
        raise FileNotFoundError(f"在 {train_img_dir} 中未找到训练图像 (*.jpg)。")
    if len(all_train_images) != len(all_train_masks):
        raise ValueError(
            f"训练图像 ({len(all_train_images)}) 与掩码 ({len(all_train_masks)}) 数量不一致。"
        )

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    available_models = {
        "UNet": UNet,
        "UNetPlusPlus": UNetPlusPlus,
        "HMTUNet": HMTUNet,
    }

    experiments = cfg.get("experiments", [])
    if not experiments:
        raise ValueError("配置文件中缺少 'experiments' 列表。")

    for experiment in experiments:
        exp_name = experiment["experiment_name"]
        model_name = experiment["model_name"]
        use_aug = experiment.get("use_augmentation", True)

        if model_name not in available_models:
            print(f"[警告] 实验 '{exp_name}' 指定的模型 '{model_name}' 不可用，已跳过。")
            continue

        print(f"\n{'=' * 25} 开始实验: {exp_name} {'=' * 25}")
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(all_train_images)):
            print(f"\n----- 第 {fold + 1}/{k_folds} 折 -----")

            train_img_paths = [all_train_images[i] for i in train_ids]
            train_mask_paths = [all_train_masks[i] for i in train_ids]
            val_img_paths = [all_train_images[i] for i in val_ids]
            val_mask_paths = [all_train_masks[i] for i in val_ids]

            train_transform, val_transform = get_transforms(img_size, augment=use_aug)

            train_dataset = ISICDataset(train_img_paths, train_mask_paths, transform=train_transform)
            val_dataset = ISICDataset(val_img_paths, val_mask_paths, transform=val_transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=hp.get("num_workers", 4),
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=max(1, hp.get("num_workers", 4) // 2),
            )

            model = available_models[model_name](
                n_channels=hp.get("in_channels", 3),
                n_classes=hp.get("num_classes", 1),
            ).to(device)

            criterion = lambda pred, target: (
                0.5 * nn.BCEWithLogitsLoss()(pred, target)
                + 0.5 * DiceLoss()(pred, target)
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            best_val_dice = 0.0
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())

            for epoch in range(1, epochs + 1):
                epoch_info = f"[{exp_name} | Fold {fold + 1}/{k_folds} | Epoch {epoch}/{epochs}]"
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, epoch_info
                )
                val_dice = evaluate(model, val_loader, device, epoch_info)
                scheduler.step()

                print(
                    f"Epoch {epoch:03d} Summary | Train Loss: {train_loss:.4f} | "
                    f"Val Dice: {val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}"
                )

                if val_dice > best_val_dice:
                    print(
                        f"  Validation Dice Improved ({best_val_dice:.4f} --> {val_dice:.4f}). Saving Model..."
                    )
                    best_val_dice = val_dice
                    best_model_weights = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                    break

            print(f"Fold {fold + 1} finished. Best Validation Dice: {best_val_dice:.4f}")
            fold_results.append(best_val_dice)

            save_path = os.path.join(
                model_save_root, f"{exp_name}_fold{fold + 1}_best.pth"
            )
            torch.save(best_model_weights, save_path)
            print(f"Best model saved to: {save_path}")

        avg_dice = float(np.mean(fold_results))
        std_dice = float(np.std(fold_results))
        print(
            f"\n{'=' * 25} {exp_name} K-Fold Cross-Validation Summary {'=' * 25}"
        )
        print(f"{k_folds}-Fold Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2017 皮肤病变分割训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
