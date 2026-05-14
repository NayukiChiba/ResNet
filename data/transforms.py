# data/transforms.py

"""
数据预处理与数据增强变换模块

功能:
1. 提供 CIFAR-10 / CIFAR-100 的标准化参数
2. 构建训练和测试用的 transform pipeline
"""

from torchvision import transforms

# CIFAR-100 标准化参数
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def buildTrainTransform(
    mean: tuple = CIFAR100_MEAN,
    std: tuple = CIFAR100_STD,
    cropSize: int = 32,
    padding: int = 4,
) -> transforms.Compose:
    """
    构建训练数据变换 pipeline(含数据增强)
    """
    return transforms.Compose(
        [
            transforms.RandomCrop(cropSize, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def buildTestTransform(
    mean: tuple = CIFAR100_MEAN,
    std: tuple = CIFAR100_STD,
) -> transforms.Compose:
    """
    构建测试数据变换 pipeline(无数据增强)
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
