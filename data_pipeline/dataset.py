"""
数据集定义与数据增强。

提供针对 ISIC 2017 皮肤病变分割任务的数据集类与变换流水线，
兼容 Albumentations 增强库与标准 torchvision transforms。
"""

import os
from typing import Tuple, Optional, Callable, List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ISICDataset(Dataset):
    """
    基于 Albumentations 的 ISIC 训练/验证数据集。

    Args:
        image_paths: 图像文件路径列表。
        mask_paths: 掩码文件路径列表，顺序需与 ``image_paths`` 一一对应。
        transform: Albumentations 变换对象，可为 ``None``。
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"图像数量 ({len(image_paths)}) 与掩码数量 ({len(mask_paths)}) 不一致。"
            )
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"图像文件未找到: {img_path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"掩码文件未找到: {mask_path}")

        # OpenCV 读取 BGR 图像并转 RGB
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"无法读取图像文件: {img_path} (cv2.imread 返回 None)")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"无法读取掩码文件: {mask_path} (cv2.imread 返回 None)")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 确保掩码为二值浮点张量并增加通道维度 (1, H, W)
        mask = (mask > 0.5).float().unsqueeze(0)
        return image, mask


class TestDataset(Dataset):
    """
    基于 torchvision transforms 的测试数据集。

    **注意**：掩码不会经过与图像相同的归一化变换，仅做尺寸调整与 ToTensor。

    Args:
        image_dir: 测试图像目录。
        mask_dir: 测试掩码目录。
        img_size: 目标尺寸 (H, W)。
        transform: 可选的外部变换；若为 ``None`` 则内部默认构建 ``Resize + ToTensor``。
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
    ) -> None:
        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"图像目录未找到: {image_dir}")
        if not os.path.isdir(mask_dir):
            raise NotADirectoryError(f"掩码目录未找到: {mask_dir}")

        self.image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
        )
        self.img_size = img_size
        self.transform = transform

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"图像数量 ({len(self.image_paths)}) 与掩码数量 ({len(self.mask_paths)}) 不一致。"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"图像文件未找到: {img_path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"掩码文件未找到: {mask_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            raise IOError(f"读取文件失败 ({img_path} / {mask_path}): {e}")

        # 图像变换：Resize + ToTensor（归一化到 [0,1]）
        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision import transforms

            image_tf = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                ]
            )
            image = image_tf(image)

        # 掩码变换：仅 Resize + ToTensor，**不做** Normalize
        mask_tf = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )
        mask = mask_tf(mask)

        return image, mask


def get_transforms(
    img_size: int, augment: bool = True
) -> Tuple[A.Compose, A.Compose]:
    """
    获取训练/验证数据增强流水线。

    Args:
        img_size: 目标图像尺寸（正方形边长）。
        augment: 训练集是否启用数据增强。

    Returns:
        (train_transform, val_transform) 元组。
    """
    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    if augment:
        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-45, 45),
                    p=0.5,
                ),
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
                A.GridDistortion(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.5),
                A.GaussNoise(p=0.2),
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ]
        )
    else:
        train_transform = val_transform

    return train_transform, val_transform
