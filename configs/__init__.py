"""
配置加载模块，提供统一的 YAML 配置读取接口。
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    加载 YAML 配置文件并返回字典。

    Args:
        config_path: 配置文件路径，默认为 ``configs/config.yaml``。

    Returns:
        包含所有配置项的字典。

    Raises:
        FileNotFoundError: 当配置文件不存在时抛出。
        yaml.YAMLError: 当 YAML 解析失败时抛出。
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件未找到: {os.path.abspath(config_path)}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件为空或解析失败: {config_path}")

    return config
