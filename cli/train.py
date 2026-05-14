"""
训练子命令
"""

from torch.optim.lr_scheduler import CosineAnnealingLR

from config.defaults import DefaultParams
from config.training import TrainingParams
from data.dataLoader import buildLoaders
from models import buildDenseNet, resnet18
from training import AdamW, CrossEntropyLoss, Trainer


def register_train_parser(subparsers):
    parser = subparsers.add_parser("train", help="训练模型")
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
    parser.add_argument("--epochs", type=int, default=TrainingParams.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=TrainingParams.LEARNING_RATE)
    parser.add_argument(
        "--weight-decay", type=float, default=TrainingParams.WEIGHT_DECAY
    )
    parser.add_argument("--device", type=str, default=DefaultParams.DEVICE)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)


def _run(args):
    device = args.device
    num_classes = 100 if args.dataset == "cifar100" else 10

    train_loader, val_loader = buildLoaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )

    if args.model == "resnet18":
        model = resnet18(num_classes=num_classes, cifar=True)
    else:
        model = buildDenseNet(num_classes=num_classes, cifar=True)

    criterion = CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name=args.model,
        epochs=args.epochs,
        device=device,
        use_early_stopping=args.early_stopping,
    )
    trainer.fit()
