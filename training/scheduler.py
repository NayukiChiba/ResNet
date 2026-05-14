"""
学习率调度器模块
"""

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts as _CosineAnnealingWarmRestarts,
)
from torch.optim.lr_scheduler import MultiStepLR as _MultiStepLR
from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR as _StepLR


class _BaseScheduler:
    """调度器基类，封装 PyTorch 调度器的通用方法"""

    _needs_metric = False  # ReduceLROnPlateau 需要传入指标

    def step(self, metric: float | None = None):
        if self._needs_metric:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_last_lr(self) -> list:
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.scheduler.load_state_dict(state_dict)


class CosineAnnealingLR(_BaseScheduler):
    """余弦退火

    用法:
        scheduler = CosineAnnealingLR(optimizer, t_max=200)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        t_max: int = 200,
        eta_min: float = 0.0,
        **kwargs: Any,
    ):
        self.scheduler = _CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min, **kwargs
        )


class CosineAnnealingWarmRestarts(_BaseScheduler):
    """带热重启的余弦退火，每次重启后周期长度翻倍

    用法:
        scheduler = CosineAnnealingWarmRestarts(optimizer, t0=50, t_mult=2)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        t0: int = 50,
        t_mult: int = 2,
        eta_min: float = 0.0,
        **kwargs: Any,
    ):
        self.scheduler = _CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min, **kwargs
        )


class StepLR(_BaseScheduler):
    """固定步长衰减，每隔 step_size 个 epoch 乘 gamma

    用法:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 30,
        gamma: float = 0.1,
        **kwargs: Any,
    ):
        self.scheduler = _StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)


class MultiStepLR(_BaseScheduler):
    """多步长衰减，在 milestones 列表指定的 epoch 节点乘 gamma

    用法:
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list = None,
        gamma: float = 0.1,
        **kwargs: Any,
    ):
        self.scheduler = _MultiStepLR(
            optimizer, milestones=milestones or [60, 120, 160], gamma=gamma, **kwargs
        )


class ReduceLROnPlateau(_BaseScheduler):
    """平台期衰减，当验证指标在 patience 个 epoch 内未改善时降低学习率

    用法:
        scheduler = ReduceLROnPlateau(optimizer, patience=10)
        scheduler.step(val_loss)  # 需要传入验证指标
    """

    _needs_metric = True

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        **kwargs: Any,
    ):
        self.scheduler = _ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            **kwargs,
        )


class OneCycleLR(_BaseScheduler):
    """单周期策略，学习率先升后降

    注意: 该调度器应在每个 batch 后调用 step()，而非每个 epoch

    用法:
        scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=200, steps_per_epoch=len(train_loader))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float = 0.01,
        epochs: int = 200,
        steps_per_epoch: int = 1,
        pct_start: float = 0.3,
        **kwargs: Any,
    ):
        self.scheduler = _OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy="cos",
            **kwargs,
        )
