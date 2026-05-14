"""
DenseNet 模型实现

1. DenseLayer: 单个稠密层,包含批归一化、ReLU 激活和卷积操作.
2. DenseBlock: 稠密块, 串联多个 DenseLayer
3. Transition: 过渡层, 包含批归一化、ReLU 激活、1x1 卷积和平均池化操作.
4. denseNetBC: DenseNet 模型的实现,包含 DenseBlock 和 Transition 层.

"""

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """
    单个稠密层,包含批归一化、ReLU 激活和卷积操作.

    BottleNeck 结构:
    BN -> ReLU -> Conv(1x1): 压缩到4 x growthRate -> BN -> ReLU -> Conv(3x3): 输出 growthRate 个特征图

    """

    def __init__(self, in_channels: int, growth_rate: int, bottle_neck: bool = True):
        """
        Args:
            in_channels: 输入特征图的通道数
            growth_rate: 每个 DenseLayer 输出的特征图数量
            bottle_neck: 是否使用 BottleNeck 结构


        """

        super().__init__()

        if bottle_neck:
            # 先使用 1x1 卷积压缩特征图数量到 4 x growth_rate
            self.layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
                ),
            )
        else:
            # 直接使用 3x3 卷积
            self.layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels, growth_rate, kernel_size=3, padding=1, bias=False
                ),
            )

    def forward(self, x):
        out = self.layer(x)
        # 将输入和输出特征图进行拼接
        out = torch.cat([x, out], dim=1)
        return out


class DenseBlock(nn.Module):
    """
    稠密块, 串联多个 DenseLayer
    """

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bottle_neck: bool = True,
    ):
        super().__init__()
        layers = []

        for i in range(num_layers):
            # 每层输入通道 = 初始输入通道 + i层已经增加的通道数

            layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, bottle_neck)
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    """
    过渡层, 包含批归一化、ReLU 激活、1x1 卷积和平均池化操作.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNetBC(nn.Module):
    """
    DenseNet 模型的实现,包含 DenseBlock 和 Transition 层.

    """

    def __init__(
        self,
        growth_rate: int = 12,
        block_config: list = None,
        num_classes: int = 10,
        bottle_neck: bool = True,
        compression: float = 0.5,
        cifar: bool = True,
    ):
        """
        Args:
            growth_rate: 每个 DenseLayer 输出的特征图数量
            block_config: 每个 DenseBlock 中 DenseLayer 的数量列表, 例如 [6, 12, 24, 16]
            num_classes: 分类任务的类别数量
            bottle_neck: 是否使用 BottleNeck 结构
            compression: 过渡层中通道数的压缩率, 例如 0.5 表示压缩到原来的一半
            cifar: 是否为 CIFAR 数据集设计网络结构, CIFAR 数据集输入图像较小

        """
        super().__init__()
        if block_config is None:
            block_config = [16, 16, 16]  # DenseNet-BC-100 的配置

        if cifar:
            # CIFAR: 3x3 卷积, 不下采样
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, growth_rate * 2, kernel_size=3, padding=1, bias=False)
            )

        else:
            # ImageNet: 7x7 卷积, 下采样
            self.layer1 = nn.Sequential(
                nn.Conv2d(
                    3, growth_rate * 2, kernel_size=7, stride=2, padding=3, bias=False
                ),
                nn.BatchNorm2d(growth_rate * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        num_channels = growth_rate * 2  # 初始通道数

        # 构建稠密块和过渡层
        blocks = []
        for i, num_layers in enumerate(block_config):
            blocks.append(
                DenseBlock(num_layers, num_channels, growth_rate, bottle_neck)
            )
            num_channels += num_layers * growth_rate  # 更新通道数

            if i != len(block_config) - 1:  # 最后一个块后不添加过渡层
                out_channels = int(num_channels * compression)  # 压缩通道数
                blocks.append(Transition(num_channels, out_channels))
                num_channels = out_channels  # 更新通道数

        self.features = nn.Sequential(*blocks)

        # 分类器
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes),
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


def densenet_bc_100(
    growth_rate: int = 12,
    num_blocks: int = 3,
    num_classes: int = 10,
    cifar: bool = True,
) -> DenseNetBC:
    """
    构建 DenseNet-BC-100 模型

    BC = BottleNeck + Compression, 默认 k = 12, depth = 100

    Args:
        growth_rate: 每个 DenseLayer 输出的特征图数量
        num_blocks: DenseBlock 的数量, 默认为 3
        num_classes: 分类任务的类别数量
        cifar: 是否为 CIFAR 数据集设计网络结构, CIFAR 数据集输入图像较小

    """
    # 每层bottle_neck有2个conv, 每个transition有1个conv
    # depth = 2 * num_blocks * num_layers_per_block + (num_blocks - 1) + (num_blocks - 1) (过渡层)
    # 解出来 num_layers_per_block = (depth - 2 * num_blocks + 1) // (2 * num_blocks)
    num_layers_per_block = (100 - 2 * num_blocks + 1) // (2 * num_blocks)

    block_config = [num_layers_per_block] * num_blocks
    return DenseNetBC(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=num_classes,
        bottle_neck=True,
        cifar=cifar,
    )
