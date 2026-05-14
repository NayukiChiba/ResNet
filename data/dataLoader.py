# data/dataLoader.py

"""
数据加载模块

功能：
1. 构建 CIFAR-10 / CIFAR-100 数据集
2. 构建训练和测试用的 DataLoader
"""

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import VisionDataset

from config.paths import DATASET_DIR
from data.transforms import buildTestTransform, buildTrainTransform

# 数据集名称到 torchvision 类的映射
_DATASET_MAP = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


def buildDatasets(
    datasetName: str = "cifar100",
    dataDir: Path = DATASET_DIR,
) -> Tuple[VisionDataset, VisionDataset]:
    """
    构建 CIFAR 训练集和测试集

    Args:
        datasetName: 数据集名称，支持 "cifar10" 和 "cifar100"
        dataDir: 数据集存储根目录

    Returns:
        (trainDataset, testDataset)
    """
    if datasetName not in _DATASET_MAP:
        raise ValueError(
            f"不支持的数据集: {datasetName}，可选: {list(_DATASET_MAP.keys())}"
        )

    datasetClass = _DATASET_MAP[datasetName]

    trainDataset = datasetClass(
        root=dataDir,
        train=True,
        download=True,
        transform=buildTrainTransform(),
    )
    testDataset = datasetClass(
        root=dataDir,
        train=False,
        download=True,
        transform=buildTestTransform(),
    )
    return trainDataset, testDataset


def buildLoaders(
    dataset: str = "cifar100",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    dataset_dir: Path = DATASET_DIR,
) -> Tuple[DataLoader, DataLoader]:
    """
    构建 CIFAR 训练和测试 DataLoader

    Args:
        dataset: 数据集名称，支持 "cifar10" 和 "cifar100"
        batch_size: 批大小
        num_workers: 数据加载子进程数
        pin_memory: 是否将数据固定到 GPU 内存
        dataset_dir: 数据集存储根目录

    Returns:
        (trainLoader, testLoader)
    """
    trainDataset, testDataset = buildDatasets(
        datasetName=dataset,
        dataDir=dataset_dir,
    )

    trainLoader = DataLoader(
        dataset=trainDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testLoader = DataLoader(
        dataset=testDataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return trainLoader, testLoader
