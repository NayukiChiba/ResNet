"""
评估指标:Top-k 准确率、混淆矩阵、各类别准确率、精确率、召回率
"""

import numpy as np
import torch


def topk_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)
) -> list:
    """计算 Top-k 准确率

    Args:
        output: 模型输出 logits,shape (batch, num_classes)
        target: 标签,shape (batch,)
        topk: 要计算的 k 值元组,默认 (1, 5)

    Returns:
        [top1_acc, top5_acc, ...],每个都是 0~1 之间的 float
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        result.append((correct_k / batch_size).item())
    return result


def confusion_matrix(
    output: torch.Tensor, target: torch.Tensor, num_classes: int
) -> np.ndarray:
    """累积更新混淆矩阵

    Args:
        output: 模型输出 logits,shape (batch, num_classes)
        target: 标签,shape (batch,)
        num_classes: 类别总数

    Returns:
        混淆矩阵 (num_classes, num_classes),行=真实标签,列=预测标签
    """
    pred = output.argmax(dim=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target.cpu().tolist(), pred.cpu().tolist()):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray, class_names: list | None = None) -> dict:
    """从混淆矩阵计算每个类别的准确率

    Args:
        cm: 混淆矩阵 (num_classes, num_classes)
        class_names: 类别名称列表,None 则用数字索引

    Returns:
        {class_name: accuracy}
    """
    num_classes = cm.shape[0]
    acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)

    if class_names is not None:
        return {name: acc[i] for i, name in enumerate(class_names)}
    return {i: acc[i] for i in range(num_classes)}


def per_class_precision(cm: np.ndarray, class_names: list | None = None) -> dict:
    """从混淆矩阵计算每个类别的精确率

    精确率 = TP / (TP + FP),即该类被预测正确的比例(按列计算)
    """
    num_classes = cm.shape[0]
    precision = cm.diagonal() / (cm.sum(axis=0) + 1e-8)

    if class_names is not None:
        return {name: precision[i] for i, name in enumerate(class_names)}
    return {i: precision[i] for i in range(num_classes)}


def per_class_recall(cm: np.ndarray, class_names: list | None = None) -> dict:
    """从混淆矩阵计算每个类别的召回率

    召回率 = TP / (TP + FN),即实际为该类的样本中被正确识别的比例(按行计算)
    """
    return per_class_accuracy(cm, class_names)
