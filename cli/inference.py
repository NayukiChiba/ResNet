"""
推理子命令
"""

import torch
from PIL import Image

from config.defaults import DefaultParams
from inference import Inferencer
from models import buildDenseNet, resnet18


def register_inference_parser(subparsers):
    parser = subparsers.add_parser("inference", help="单张图片推理")
    parser.set_defaults(func=_run)

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "densenet"],
        help="模型名称",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument("--topk", type=int, default=5, help="显示 Top-K 结果")
    parser.add_argument("--device", type=str, default=DefaultParams.DEVICE)


def _run(args):
    device = args.device

    # 从检查点推断类别数（找最后一个 Linear 层的权重形状）
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    num_classes = None
    for key in reversed(list(state_dict.keys())):
        if key.endswith(".weight") and state_dict[key].ndim == 2:
            num_classes = state_dict[key].size(0)
            break
    if num_classes is None:
        raise ValueError("无法从检查点推断类别数")

    if args.model == "resnet18":
        model = resnet18(num_classes=num_classes, cifar=True)
    else:
        model = buildDenseNet(num_classes=num_classes, cifar=True)
    model.load_state_dict(state_dict)

    inferencer = Inferencer(model, class_names=None, device=device)
    image = Image.open(args.image).convert("RGB")
    results = inferencer.predict_topk(image, k=args.topk)

    print(f"Image: {args.image}")
    for i, (label, conf) in enumerate(results, 1):
        print(f"  Top-{i}: class={label}, confidence={conf:.4f}")
