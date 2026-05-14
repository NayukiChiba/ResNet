"""
可视化模块:训练曲线、混淆矩阵
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict, save_path: str | None = None):
    """绘制训练曲线(loss + accuracy 双子图)

    Args:
        history: {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}
        save_path: 保存路径,None 则仅显示
    """
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
):
    """绘制混淆矩阵(原始计数 + 归一化)

    Args:
        cm: 混淆矩阵 (num_classes, num_classes)
        class_names: 类别名称列表,类别数超过 20 时自动隐藏标签
        title: 图表标题
        save_path: 保存路径,None 则仅显示
    """
    num_classes = cm.shape[0]
    cm_normalized = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"{title} - Count", fontsize=14)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"{title} - Normalized", fontsize=14)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.colorbar(im1, ax=axes[1])

    if class_names is not None and num_classes <= 20:
        for ax in axes:
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=90, fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
