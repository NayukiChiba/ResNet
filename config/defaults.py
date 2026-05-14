"""
全局默认常量：随机种子、计算设备

用法:
    from config.defaults import DefaultParams
    from config import DefaultParams
"""

import torch


class DefaultParams:
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
