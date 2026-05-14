"""
损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """交叉熵损失,支持标签平滑

    用法:
        criterion = CrossEntropyLoss(label_smoothing=0.1)
        loss = criterion(outputs, labels)
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(inputs, targets)


class FocalLoss(nn.Module):
    """Focal Loss,缓解类别不平衡

    公式: FL = -(1 - p_t)^\gamma * log(p_t)

    用法:
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(outputs, labels)
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(inputs, targets, reduction="none")
        prob = torch.exp(-cross_entropy)
        focal_loss = (1 - prob) ** self.gamma * cross_entropy
        if self.alpha is not None:
            focal_loss = self.alpha.to(inputs.device)[targets] * focal_loss
        return focal_loss.mean()
