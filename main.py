"""
CLI 入口：训练 / 评估 / 推理

用法:
    python main.py train --model resnet18 --dataset cifar100
    python main.py evaluate --model resnet18 --checkpoint best.pth
    python main.py inference --model resnet18 --checkpoint best.pth --image cat.jpg
"""

import argparse

from cli.evaluate import register_evaluate_parser
from cli.inference import register_inference_parser
from cli.train import register_train_parser


def main():
    parser = argparse.ArgumentParser(
        prog="resnet", description="ResNet 模型训练与推理工具"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_train_parser(subparsers)
    register_evaluate_parser(subparsers)
    register_inference_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
