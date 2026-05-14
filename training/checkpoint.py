"""
检查点模块

每次保存会写入两个文件:
- last.pth: 最近一次保存的检查点(断点续训用)
- best.pth: 验证集指标最优时的检查点(最终推理用)

保存内容:模型权重、优化器状态、调度器状态、当前 epoch、最佳指标值
"""

from pathlib import Path

import torch
import torch.nn as nn

from config.paths import CHECKPOINTS_DIR


class Checkpoint:
    """模型检查点保存与加载

    用法:
        ckpt = Checkpoint("resnet18")

        # 每个 epoch 结束后保存
        ckpt.save(model, optimizer, scheduler, epoch=epoch, best_metric=val_acc, is_best=True)

        # 恢复训练或推理
        epoch, best_metric = ckpt.load(model, optimizer, scheduler, best=True)
    """

    def __init__(self, model_name: str, save_dir: Path = CHECKPOINTS_DIR):
        # 每个模型一个子目录,避免不同模型的检查点混在一起
        self.model_name = model_name
        self.save_dir = Path(save_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 最佳模型和最新模型的保存路径
        self.best_path = self.save_dir / "best.pth"
        self.last_path = self.save_dir / "last.pth"

    def save(
        self,
        model: nn.Module,
        optimizer=None,
        scheduler=None,
        epoch: int = 0,
        best_metric: float = 0.0,
        is_best: bool = False,
    ):
        """保存检查点

        Args:
            model: 模型实例
            optimizer: 优化器实例,None 则不保存优化器状态
            scheduler: 调度器实例,None 则不保存调度器状态
            epoch: 当前 epoch 编号
            best_metric: 当前最佳验证指标值
            is_best: 是否为当前最佳模型,True 时额外写入 best.pth
        """
        # 组装要保存的状态字典,至少包含模型权重和 epoch
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_metric": best_metric,
        }

        # 如果传入了优化器/调度器,一并保存它们的状态以便断点续训
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        # last.pth 每个 epoch 都覆盖写入
        torch.save(state, self.last_path)

        # best.pth 只在验证指标改善时写入
        if is_best:
            torch.save(state, self.best_path)

    def load(
        self, model: nn.Module, optimizer=None, scheduler=None, best: bool = False
    ):
        """加载检查点

        Args:
            model: 模型实例,直接原地加载权重
            optimizer: 优化器实例,传入则恢复优化器状态
            scheduler: 调度器实例,传入则恢复调度器状态
            best: True 加载 best.pth,False 加载 last.pth

        Returns:
            (epoch, best_metric): 恢复后的 epoch 编号和最佳指标值
        """
        path = self.best_path if best else self.last_path

        if not path.exists():
            raise FileNotFoundError(f"检查点不存在: {path}")

        # weights_only=False 允许加载非张量数据(epoch、config 等)
        state = torch.load(path, map_location="cpu", weights_only=False)

        # 加载模型权重
        model.load_state_dict(state["model_state_dict"])

        # 仅在传入了优化器且检查点包含其状态时才恢复
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # 同上,按需恢复调度器状态
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        # 返回 epoch 和最佳指标,方便调用方继续训练或记录
        return state.get("epoch", 0), state.get("best_metric", 0.0)
