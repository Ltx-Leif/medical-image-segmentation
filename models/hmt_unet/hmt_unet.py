"""
HMT-UNet 模型封装。

本模块对原始 MambaVision_sim 分割头进行包装，
统一对外接口以兼容 ``n_channels`` / ``n_classes`` 命名规范。
"""

from typing import Optional
import torch
from torch import nn

from .transformer import MambaVision_sim


class HMTUNet(nn.Module):
    """
    HMT-UNet 分割网络。

    基于 MambaVision 编码器结构，融合 Transformer 与 CNN 特征，
    适用于高分辨率医学图像分割任务。

    Args:
        n_channels: 输入图像通道数，默认 3。
        n_classes: 输出类别数，二分类时默认 1。
            无论 ``n_classes`` 取值如何，本封装均统一对外输出 **logits**，
            使训练、推理流程与 UNet / UNet++ 完全一致。
        depths: 每个阶段的层数列表。
        num_heads: 每个阶段的注意力头数列表。
        window_size: 每个阶段的窗口大小列表。
        dim: 初始特征维度。
        in_dim: PatchEmbed 后的初始维度。
        mlp_ratio: MLP 隐藏层倍数。
        resolution: 模型设计分辨率。
        drop_path_rate: DropPath 正则化比率。
        load_ckpt_path: 可选的预训练权重路径。
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        depths: list = [1, 3, 8, 4],
        num_heads: list = [2, 4, 8, 16],
        window_size: list = [8, 8, 14, 7],
        dim: int = 80,
        in_dim: int = 32,
        mlp_ratio: int = 4,
        resolution: int = 224,
        drop_path_rate: float = 0.2,
        load_ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = n_classes
        self.load_ckpt_path = load_ckpt_path

        self.hmtunet = MambaVision_sim(
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            dim=dim,
            in_dim=in_dim,
            mlp_ratio=mlp_ratio,
            resolution=resolution,
            drop_path_rate=drop_path_rate,
            in_chans=n_channels,
            num_classes=n_classes,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"HMTUNet 期望输入 4D 张量 (B, C, H, W)，但获得 {x.dim()}D"
            )
        out = self.hmtunet(x)
        # 统一对外接口：当 num_classes==1 时，内部 MambaVision_sim 已做 sigmoid，
        # 此处通过 logit 反变换将概率映射回 logits，使外部可统一使用 torch.sigmoid。
        if self.num_classes == 1:
            eps = 1e-6
            out = torch.clamp(out, eps, 1.0 - eps)
            out = torch.log(out) - torch.log1p(-out)
        return out

    def load_from(self) -> None:
        """
        从 ``load_ckpt_path`` 加载预训练权重，分别处理 encoder 与 decoder 键名映射。
        """
        if self.load_ckpt_path is None:
            print("未提供 load_ckpt_path，跳过预训练权重加载。")
            return

        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            map_location = "cpu"
        else:
            map_location = None

        # Encoder
        model_dict = self.hmtunet.state_dict()
        checkpoint = torch.load(self.load_ckpt_path, map_location=map_location)
        pretrained_dict = checkpoint.get("state_dict", checkpoint)

        new_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
        }
        model_dict.update(new_dict)
        self.hmtunet.load_state_dict(model_dict)
        not_loaded = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
        print(
            f"[Encoder] Total: {len(model_dict)}, Pretrained: {len(pretrained_dict)}, "
            f"Updated: {len(new_dict)}, Not loaded: {not_loaded}"
        )

        # Decoder: 将 levels.x 映射为 layers_up.(3-x)
        pretrained_order_dict = checkpoint.get("state_dict", checkpoint)
        decoder_dict: dict = {}
        for k, v in pretrained_order_dict.items():
            if "levels.0" in k:
                decoder_dict[k.replace("levels.0", "layers_up.3")] = v
            elif "levels.1" in k:
                decoder_dict[k.replace("levels.1", "layers_up.2")] = v
            elif "levels.2" in k:
                decoder_dict[k.replace("levels.2", "layers_up.1")] = v
            elif "levels.3" in k:
                decoder_dict[k.replace("levels.3", "layers_up.0")] = v

        model_dict = self.hmtunet.state_dict()
        new_dict = {k: v for k, v in decoder_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        self.hmtunet.load_state_dict(model_dict)
        not_loaded = [k for k in decoder_dict.keys() if k not in new_dict.keys()]
        print(
            f"[Decoder] Total: {len(model_dict)}, Pretrained: {len(decoder_dict)}, "
            f"Updated: {len(new_dict)}, Not loaded: {not_loaded}"
        )
