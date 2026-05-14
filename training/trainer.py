"""
训练器模块
"""

from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.checkpoint import Checkpoint
from training.early_stopping import EarlyStopping
from training.logger import Logger


class Trainer:
    """训练器

    用法:
        trainer = Trainer(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            model_name="resnet18",
            epochs=20,
            device="cuda",
        )
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer,
        scheduler=None,
        model_name: str = "model",
        epochs: int = 20,
        device: str = "cuda",
        use_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.epochs = epochs
        self.device = device

        self.logger = Logger(model_name)
        self.checkpoint = Checkpoint(model_name)

        self.early_stopping = None
        if use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )

        self.best_metric = 0.0
        self.current_epoch = 0

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._evaluate(epoch, "val")

            is_best = val_acc > self.best_metric
            if is_best:
                self.best_metric = val_acc

            self.checkpoint.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=epoch,
                best_metric=self.best_metric,
                is_best=is_best,
            )

            self.logger.log(train_loss, train_acc, val_loss, val_acc)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if self.early_stopping is not None and self.early_stopping(val_loss):
                print(f"早停触发于 epoch {epoch}")
                break

        self.logger.plot()

    def test(self, test_loader: DataLoader):
        self.model.load_state_dict(
            torch.load(
                self.checkpoint.best_path, map_location=self.device, weights_only=False
            )["model_state_dict"]
        )
        test_loss, test_acc = self._evaluate(0, "test", test_loader)
        print(f"Test  | loss {test_loss:.4f} | acc {test_acc:.4f}")
        return test_loss, test_acc

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[Train] Epoch {epoch:03d}",
            leave=False,
            ascii=False,
        )

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        return running_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(
        self, epoch: int, task: Literal["val", "test"], loader: DataLoader | None = None
    ):
        self.model.eval()
        loader = loader or self.val_loader
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            loader,
            desc=f"[{task.capitalize()}] Epoch {epoch:03d}",
            leave=False,
            ascii=False,
        )

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        return running_loss / total, correct / total
