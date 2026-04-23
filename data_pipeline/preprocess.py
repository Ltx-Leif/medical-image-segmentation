"""
ISIC 2017 数据集预处理脚本。

将原始 ISIC 数据集统一尺寸、二值化掩码，并按 ``train/val/test`` 结构输出。
"""

import os
import glob
from typing import Tuple
import cv2
import numpy as np
from tqdm import tqdm


def process_isic_data(
    root_path: str,
    output_path: str,
    img_size: Tuple[int, int] = (256, 256),
) -> None:
    """
    对 ISIC 2017 数据集进行预处理。

    Args:
        root_path: 包含原始 ISIC 数据集文件夹的根目录。
        output_path: 处理后数据的保存目录。
        img_size: 目标图像和掩码的尺寸，默认为 ``(256, 256)``。

    Raises:
        NotADirectoryError: 当 ``root_path`` 不存在时抛出。
    """
    if not os.path.isdir(root_path):
        raise NotADirectoryError(f"原始数据根目录不存在: {root_path}")

    print(f"开始预处理 ISIC 2017 数据集，目标尺寸: {img_size}")

    data_map = [
        ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth", "train"),
        ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth", "val"),
        ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth", "test"),
    ]

    for img_folder_name, mask_folder_name, split_name in data_map:
        print(f"\n===== 正在处理 {split_name} 数据集 =====")

        img_folder = os.path.join(root_path, img_folder_name)
        mask_folder = os.path.join(root_path, mask_folder_name)

        if not os.path.isdir(img_folder):
            print(f"[跳过] 图像文件夹不存在: {img_folder}")
            continue
        if not os.path.isdir(mask_folder):
            print(f"[跳过] 掩码文件夹不存在: {mask_folder}")
            continue

        image_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
        if not image_paths:
            print(f"[跳过] 在 {img_folder} 中未找到 *.jpg 图像。")
            continue

        output_img_folder = os.path.join(output_path, split_name, "images")
        output_mask_folder = os.path.join(output_path, split_name, "masks")
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_mask_folder, exist_ok=True)

        for img_path in tqdm(image_paths, desc=f"处理 {split_name} 图像"):
            try:
                base_name = os.path.basename(img_path).replace(".jpg", "")
                mask_name = f"{base_name}_segmentation.png"
                mask_path = os.path.join(mask_folder, mask_name)

                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"[警告] 无法读取图像: {img_path}")
                    continue
                if mask is None:
                    print(f"[警告] 无法读取掩码: {mask_path}")
                    continue

                image_resized = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

                _, mask_binary = cv2.threshold(mask_resized, 128, 1, cv2.THRESH_BINARY)

                out_img_path = os.path.join(output_img_folder, os.path.basename(img_path))
                out_mask_path = os.path.join(output_mask_folder, mask_name)

                cv2.imwrite(out_img_path, image_resized)
                cv2.imwrite(
                    out_mask_path, (mask_binary.astype(np.uint8) * 255)
                )
            except Exception as e:
                print(f"[错误] 处理 {img_path} 时发生异常: {e}")

    print("\n数据预处理完成！输出目录:", os.path.abspath(output_path))


if __name__ == "__main__":
    RAW_DATA_ROOT = "./data"
    PROCESSED_DATA_PATH = "./data/processed"
    TARGET_SIZE: Tuple[int, int] = (256, 256)

    process_isic_data(
        root_path=RAW_DATA_ROOT,
        output_path=PROCESSED_DATA_PATH,
        img_size=TARGET_SIZE,
    )
