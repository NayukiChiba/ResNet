"""
ResNet 模型实现

1. BasicBlock 实现
2. ResNet主模型
3. resNet18 构建函数
"""

import torch
from torch import nn


class BasicBlock(nn.Module):
    """
    ResNet 基础块实现
    2 层 3x3 卷积 + BatchNorm + ReLU

    """

    expansion = 1  # 扩展系数，BasicBlock 不改变通道数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample  # 下采样层，用于调整维度

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)  # 调整输入维度以匹配输出
        out += identity  # 残差连接
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet 主模型实现
    """

    def __init__(
        self, block: nn.Module, layers: list, num_classes: int = 100, cifar: bool = True
    ):
        super().__init__()

        self.in_channels = 64

        if cifar:
            # 如果是 CIFAR 数据集，使用 3x3 卷积和适当的池化
            self.conv1 = nn.Conv2d(
                3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()  # CIFAR 不需要池化
        else:
            # ImageNet 数据集使用 7x7 卷积和池化
            self.conv1 = nn.Conv2d(
                3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # stride 为 1 的层不改变空间尺寸，stride 为 2 的层会将空间尺寸减半
        self.layer1 = self._make_layer(
            block=block, out_channels=64, blocks=layers[0], stride=1
        )
        # stride 为 2, 会下采样，空间尺寸减半，通道数翻倍
        self.layer2 = self._make_layer(
            block=block, out_channels=128, blocks=layers[1], stride=2
        )
        # 继续下采样，空间尺寸减半，通道数翻倍
        self.layer3 = self._make_layer(
            block=block, out_channels=256, blocks=layers[2], stride=2
        )
        # 继续下采样，空间尺寸减半，通道数翻倍
        self.layer4 = self._make_layer(
            block=block, out_channels=512, blocks=layers[3], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: nn.Module, out_channels: int, blocks: int, stride: int = 1
    ):
        """构建一个残差层"""
        downsample = None
        # 如果 stride 不为 1 或输入通道数不匹配输出通道数，需要下采样调整维度
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        # 如果 stride 为 1 且输入通道数匹配输出通道数，则不需要下采样
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        # 更新输入通道数为输出通道数 * 扩展系数，以便后续块使用
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=10, cifar=True):
    """构建 ResNet-18 模型"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar=cifar)
