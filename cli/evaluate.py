"""
评估子命令
"""

import torch

from config.defaults import DefaultParams
from data.dataLoader import buildLoaders
from evaluation import Evaluator
from models import buildDenseNet, resnet18


def register_evaluate_parser(subparsers):
    parser = subparsers.add_parser("evaluate", help="在测试集上评估模型")
    parser.set_defaults(func=_run)

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "densenet"],
        help="模型名称",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100"],
        help="数据集",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=DefaultParams.DEVICE)
    parser.add_argument("--save-cm", type=str, default=None, help="混淆矩阵保存路径")


def _run(args):
    device = args.device
    num_classes = 100 if args.dataset == "cifar100" else 10

    _, test_loader = buildLoaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )

    if args.model == "resnet18":
        model = resnet18(num_classes=num_classes, cifar=True)
    else:
        model = buildDenseNet(num_classes=num_classes, cifar=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(model, test_loader, device=device)
    evaluator.evaluate()
    evaluator.plot_confusion_matrix(save_path=args.save_cm)
