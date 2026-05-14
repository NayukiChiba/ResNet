"""
评估器：在验证集/测试集上运行评估
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import (
    confusion_matrix,
    per_class_accuracy,
    plot_confusion_matrix,
    topk_accuracy,
)


class Evaluator:
    """验证集评估器

    用法:
        evaluator = Evaluator(model, test_loader, criterion, class_names, device="cuda")
        results = evaluator.evaluate()
        evaluator.plot_confusion_matrix()
        per_class = evaluator.per_class_accuracy()
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module | None = None,
        class_names: list | None = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.loader = loader
        self.criterion = criterion
        self.class_names = class_names
        self.device = device

        self.num_classes = None
        self.loss: float | None = None
        self.top1: float | None = None
        self.top5: float | None = None
        self.cm: np.ndarray | None = None

    @torch.no_grad()
    def evaluate(self) -> dict:
        """运行评估，返回指标字典

        Returns:
            {"loss": ..., "top1": ..., "top5": ...}
        """
        self.model.eval()

        # 从模型最后一层推断类别数
        dummy = next(iter(self.loader))[0][:1].to(self.device)
        self.num_classes = self.model(dummy).size(1)

        running_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total = 0
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        pbar = tqdm(self.loader, desc="Evaluate", leave=False, ascii=False)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = labels.size(0)

            outputs = self.model(images)

            if self.criterion is not None:
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * batch_size

            # top1 / top5
            t1, t5 = topk_accuracy(outputs, labels, topk=(1, 5))
            top1_correct += t1 * batch_size
            top5_correct += t5 * batch_size
            total += batch_size

            # 混淆矩阵
            self.cm += confusion_matrix(outputs, labels, self.num_classes)

            pbar.set_postfix(
                top1=f"{100.0 * top1_correct / total:.2f}%",
                top5=f"{100.0 * top5_correct / total:.2f}%",
            )

        self.top1 = top1_correct / total
        self.top5 = top5_correct / total
        self.loss = running_loss / total if self.criterion is not None else None

        print(f"Top-1: {self.top1:.4f} ({self.top1 * 100:.2f}%)")
        print(f"Top-5: {self.top5:.4f} ({self.top5 * 100:.2f}%)")
        if self.loss is not None:
            print(f"Loss:  {self.loss:.4f}")

        return {"loss": self.loss, "top1": self.top1, "top5": self.top5}

    def plot_confusion_matrix(self, save_path: str | None = None):
        """绘制混淆矩阵，需先调用 evaluate()"""
        if self.cm is None:
            raise RuntimeError("请先调用 evaluate()")
        plot_confusion_matrix(self.cm, self.class_names, "Confusion Matrix", save_path)

    def per_class_accuracy(self) -> dict:
        """计算每个类别的准确率，需先调用 evaluate()"""
        if self.cm is None:
            raise RuntimeError("请先调用 evaluate()")
        return per_class_accuracy(self.cm, self.class_names)
