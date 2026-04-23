"""
历史实验结果可视化脚本。

本脚本加载已固化的 30 组交叉验证结果（UNet / UNet++ / HMTUNet，
有/无数据增强各 5 折），绘制雷达图、平行坐标图与柱状图，
用于模型性能对比分析。

**注意**：以下 ``data`` 字典为历史实验结果硬编码，非动态计算所得。
"""

from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np


def load_best_models() -> pd.DataFrame:
    """
    构建包含历史实验结果的 DataFrame，并按 Config 分组取每组 IoU 最优的一折。

    Returns:
        每组最优模型的 DataFrame。
    """
    data = {
        "model": [
            "UNet_with_Augmentation_fold1_best.pth",
            "UNet_with_Augmentation_fold2_best.pth",
            "UNet_with_Augmentation_fold3_best.pth",
            "UNet_with_Augmentation_fold4_best.pth",
            "UNet_with_Augmentation_fold5_best.pth",
            "UNet_without_Augmentation_fold1_best.pth",
            "UNet_without_Augmentation_fold2_best.pth",
            "UNet_without_Augmentation_fold3_best.pth",
            "UNet_without_Augmentation_fold4_best.pth",
            "UNet_without_Augmentation_fold5_best.pth",
            "UNet++_with_Augmentation_fold1_best.pth",
            "UNet++_with_Augmentation_fold2_best.pth",
            "UNet++_with_Augmentation_fold3_best.pth",
            "UNet++_with_Augmentation_fold4_best.pth",
            "UNet++_with_Augmentation_fold5_best.pth",
            "UNet++_without_Augmentation_fold1_best.pth",
            "UNet++_without_Augmentation_fold2_best.pth",
            "UNet++_without_Augmentation_fold3_best.pth",
            "UNet++_without_Augmentation_fold4_best.pth",
            "UNet++_without_Augmentation_fold5_best.pth",
            "HMTUNet_with_Augmentation_fold1_best.pth",
            "HMTUNet_with_Augmentation_fold2_best.pth",
            "HMTUNet_with_Augmentation_fold3_best.pth",
            "HMTUNet_with_Augmentation_fold4_best.pth",
            "HMTUNet_with_Augmentation_fold5_best.pth",
            "HMTUNet_without_Augmentation_fold1_best.pth",
            "HMTUNet_without_Augmentation_fold2_best.pth",
            "HMTUNet_without_Augmentation_fold3_best.pth",
            "HMTUNet_without_Augmentation_fold4_best.pth",
            "HMTUNet_without_Augmentation_fold5_best.pth",
        ],
        "iou": [
            0.5157, 0.5138, 0.5637, 0.5628, 0.5777,
            0.5526, 0.4364, 0.4315, 0.4724, 0.3955,
            0.6509, 0.5825, 0.5579, 0.6551, 0.5975,
            0.3905, 0.3261, 0.2683, 0.5534, 0.4106,
            0.6158, 0.5854, 0.6014, 0.5774, 0.5461,
            0.5559, 0.5319, 0.5092, 0.4376, 0.5313,
        ],
        "dice": [
            0.6031, 0.6011, 0.6531, 0.6495, 0.6664,
            0.6524, 0.5198, 0.5127, 0.5636, 0.4738,
            0.7490, 0.6689, 0.6409, 0.7488, 0.6861,
            0.4679, 0.3903, 0.3358, 0.6510, 0.4879,
            0.7087, 0.6725, 0.6895, 0.6598, 0.6483,
            0.6604, 0.6272, 0.6116, 0.5337, 0.6289,
        ],
        "precision": [
            0.9593, 0.9766, 0.9544, 0.9503, 0.9410,
            0.9089, 0.9741, 0.9734, 0.9633, 0.9737,
            0.9060, 0.9539, 0.9680, 0.9349, 0.9337,
            0.9757, 0.9759, 0.9904, 0.9286, 0.9748,
            0.9358, 0.9371, 0.9368, 0.9297, 0.8965,
            0.9020, 0.9214, 0.9074, 0.9099, 0.9273,
        ],
        "recall": [
            0.5454, 0.5285, 0.5991, 0.5974, 0.6226,
            0.6059, 0.4478, 0.4473, 0.4902, 0.4059,
            0.7240, 0.6164, 0.5830, 0.7035, 0.6493,
            0.4018, 0.3393, 0.2726, 0.5939, 0.4224,
            0.6576, 0.6235, 0.6371, 0.6234, 0.6125,
            0.6056, 0.5663, 0.5522, 0.4794, 0.5710,
        ],
        "accuracy": [
            0.8427, 0.8483, 0.8566, 0.8574, 0.8620,
            0.8627, 0.8291, 0.8362, 0.8459, 0.8313,
            0.8861, 0.8640, 0.8621, 0.8860, 0.8667,
            0.8224, 0.8207, 0.8080, 0.8592, 0.8319,
            0.8792, 0.8707, 0.8769, 0.8671, 0.8513,
            0.8714, 0.8598, 0.8534, 0.8296, 0.8550,
        ],
        "specificity": [
            0.9969, 0.9976, 0.9950, 0.9945, 0.9933,
            0.9902, 0.9967, 0.9956, 0.9940, 0.9965,
            0.9879, 0.9935, 0.9949, 0.9900, 0.9919,
            0.9973, 0.9962, 0.9986, 0.9910, 0.9961,
            0.9891, 0.9882, 0.9889, 0.9890, 0.9898,
            0.9816, 0.9898, 0.9865, 0.9904, 0.9905,
        ],
    }
    df = pd.DataFrame(data)
    df["Config"] = df["model"].apply(
        lambda n: (
            "HMTUNet" if n.startswith("HMTUNet") else ("UNet++" if n.startswith("UNet++") else "UNet")
        )
        + (" w/ Aug" if "with_Augmentation" in n else " w/o Aug")
    )
    best = df.loc[df.groupby("Config")["iou"].idxmax()].reset_index(drop=True)
    return best


def plot_radar(df: pd.DataFrame, metrics: List[str]) -> None:
    """
    绘制雷达图（蜘蛛图）。

    Args:
        df: 包含模型配置与各指标列的 DataFrame。
        metrics: 要绘制的指标列名列表。
    """
    labels = [m.upper() for m in metrics]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for _, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["Config"], linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Radar Chart of Best Models", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


def plot_parallel(df: pd.DataFrame, metrics: List[str]) -> None:
    """
    绘制平行坐标图。

    Args:
        df: 包含模型配置与各指标列的 DataFrame。
        metrics: 要绘制的指标列名列表。
    """
    df_pc = df[["Config"] + metrics].copy()
    plt.figure(figsize=(8, 5))
    parallel_coordinates(df_pc, "Config", colormap=plt.cm.Set2)
    plt.title("Parallel Coordinates of Best Models")
    plt.ylabel("Metric Value")
    plt.legend(title="Config", loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_bar(df: pd.DataFrame, metrics: List[str], colors: List[Tuple[float, ...]]) -> None:
    """
    绘制分组柱状图。

    Args:
        df: 包含模型配置与各指标列的 DataFrame。
        metrics: 要绘制的指标列名列表。
        colors: 每组柱子的颜色列表。
    """
    labels = [m.upper() for m in metrics]
    x = np.arange(len(metrics))
    total_width = 0.8
    n = len(df)
    width = total_width / n

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in df.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width=width, label=row["Config"], color=colors[i])

    ax.set_xticks(x + total_width / 2 - width / 2)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Metric Value")
    ax.set_title("Bar Chart of Best Models", loc="left")
    ax.legend(title="Config", loc="upper right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    best = load_best_models()

    # 雷达图与平行坐标图仅使用核心指标，避免过度拥挤
    core_metrics = ["iou", "dice", "recall", "accuracy"]
    all_metrics = ["iou", "dice", "precision", "recall", "accuracy", "specificity"]

    plot_radar(best, core_metrics)
    plot_parallel(best, core_metrics)

    cmap = plt.cm.Set2
    colors = [cmap(i) for i in range(len(best))]
    plot_bar(best, all_metrics, colors)


if __name__ == "__main__":
    main()
