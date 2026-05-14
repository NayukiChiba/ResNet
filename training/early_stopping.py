"""
早停模块
"""


class EarlyStopping:
    """早停

    当验证指标在 patience 个 epoch 内未改善超过 min_delta 时触发

    用法:
        stopper = EarlyStopping(patience=10, min_delta=0.001)
        if stopper(val_loss):
            break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: float | None = None
        self.counter: int = 0

    def __call__(self, metric: float) -> bool:
        """返回 True 表示应该停止"""
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
