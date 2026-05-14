"""
优化器模块
"""

from typing import Any

from torch.optim import SGD as _SGD
from torch.optim import Adam as _Adam
from torch.optim import AdamW as _AdamW
from torch.optim import RMSprop as _RMSprop


class SGD:
    """SGD + Momentum + Nesterov

    用法:
        opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    """

    def __new__(
        cls,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ):
        return _SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
            **kwargs,
        )


class Adam:
    """Adam

    用法:
        opt = Adam(model.parameters(), lr=0.001)
    """

    def __new__(
        cls,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ):
        return _Adam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs
        )


class AdamW:
    """AdamW 解耦权重衰减

    用法:
        opt = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    """

    def __new__(
        cls,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ):
        return _AdamW(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs
        )


class RMSprop:
    """RMSprop

    用法:
        opt = RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    """

    def __new__(
        cls,
        params,
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        **kwargs: Any,
    ):
        return _RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            **kwargs,
        )
