# ResNet

基于 PyTorch 的图像分类训练框架，支持 ResNet-18 和 DenseNet-BC-100，面向 CIFAR-10/100。

## 项目结构

```
├── main.py                 # CLI 入口
├── pyproject.toml          # 项目配置与依赖
│
├── models/                 # 模型定义
│   ├── resnet.py           #   ResNet-18 (BasicBlock)
│   └── densenet.py         #   DenseNet-BC-100 (DenseBlock + Transition)
│
├── data/                   # 数据加载与预处理
│   ├── transforms.py       #   训练/测试 transform pipeline
│   └── dataLoader.py       #   CIFAR DataLoader 构建
│
├── training/               # 训练组件
│   ├── loss.py             #   CrossEntropyLoss / FocalLoss
│   ├── optimizer.py        #   SGD / Adam / AdamW / RMSprop
│   ├── scheduler.py        #   CosineAnnealingLR / StepLR / ReduceLROnPlateau 等
│   ├── early_stopping.py   #   早停
│   ├── logger.py           #   训练日志记录与曲线绘制
│   ├── checkpoint.py       #   检查点保存与加载
│   └── trainer.py          #   训练循环封装
│
├── evaluation/             # 评估与可视化
│   ├── metrics.py          #   Top-k 准确率 / 混淆矩阵 / 各类别精确率召回率
│   ├── visualization.py    #   训练曲线 / 混淆矩阵绘图
│   └── evaluator.py        #   验证集评估器
│
├── inference/              # 推理
│   └── inferencer.py       #   单张 / 批量 / Top-K 推理 / 特征提取
│
├── config/                 # 全局配置
│   ├── defaults.py         #   随机种子 / 设备
│   ├── paths.py            #   路径常量
│   ├── datasets.py         #   数据超参数
│   └── training.py         #   训练超参数
│
└── cli/                    # CLI 子命令
    ├── train.py            #   train 子命令
    ├── evaluate.py         #   evaluate 子命令
    └── inference.py        #   inference 子命令
```

## 安装

```bash
git clone <repo-url> && cd ResNet
uv sync
```

## 快速开始

### CLI

```bash
# 训练
python main.py train --model resnet18 --dataset cifar100 --epochs 20

# 评估
python main.py evaluate --model resnet18 --checkpoint outputs/checkpoints/resnet18/best.pth

# 推理
python main.py inference --model resnet18 --checkpoint outputs/checkpoints/resnet18/best.pth --image cat.jpg
```

### Python API

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import buildLoaders
from models import resnet18
from training import CrossEntropyLoss, AdamW, Trainer

train_loader, val_loader = buildLoaders("cifar100", batch_size=128)

model = resnet18(num_classes=100)
criterion = CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

trainer = Trainer(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    model_name="resnet18", epochs=200, device="cuda",
)
trainer.fit()
```

### 评估

```python
from evaluation import Evaluator

evaluator = Evaluator(model, test_loader, class_names=class_names, device="cuda")
results = evaluator.evaluate()
# Top-1: 0.7255 (72.55%)
# Top-5: 0.9201 (92.01%)

evaluator.plot_confusion_matrix(save_path="outputs/confusion_matrix.png")
per_class = evaluator.per_class_accuracy()
```

### 推理

```python
from PIL import Image
from inference import Inferencer

inferencer = Inferencer(model, class_names=class_names, device="cuda")

image = Image.open("cat.jpg").convert("RGB")
label, conf = inferencer.predict(image)
top5 = inferencer.predict_topk(image, k=5)
```

### 独立指标计算

```python
from evaluation import topk_accuracy, confusion_matrix, per_class_precision

t1, t5 = topk_accuracy(outputs, labels, topk=(1, 5))
cm = confusion_matrix(outputs, labels, num_classes=100)
precision = per_class_precision(cm, class_names)
```
