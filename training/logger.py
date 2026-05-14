"""
日志模块
"""

from pathlib import Path

import matplotlib.pyplot as plt

from config.paths import LOGS_DIR


class Logger:
    """训练日志，记录 loss 和 accuracy 并绘图

    用法:
        logger = Logger("resnet18")
        logger.log(train_loss=0.5, train_acc=0.8, val_loss=0.6, val_acc=0.75)
        logger.plot()
    """

    def __init__(self, model_name: str = "model", save_dir: Path = LOGS_DIR):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def log(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        last = {k: f"{v[-1]:.4f}" for k, v in self.history.items() if v}
        print(
            f"Epoch {len(self.history['train_loss']):03d} | "
            f"train loss {last['train_loss']} | "
            f"train acc {last['train_acc']} | "
            f"val loss {last['val_loss']} | "
            f"val acc {last['val_acc']}"
        )

    def plot(self):
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history["train_loss"], label="train")
        axes[0].plot(self.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(self.history["train_acc"], label="train")
        axes[1].plot(self.history["val_acc"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].legend()

        path = self.save_dir / f"{self.model_name}_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
