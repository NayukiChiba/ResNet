"""
推理模块
"""

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class Inferencer:
    """推理器

    用法:
        inferencer = Inferencer(model, transform, class_names, device="cuda")
        label, conf = inferencer.predict(image)
        top5 = inferencer.predict_topk(image, k=5)
    """

    def __init__(
        self,
        model: nn.Module,
        transform: transforms.Compose | None = None,
        class_names: list | None = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names

        # 默认 transform:ToTensor + 归一化(CIFAR 用,ImageNet 需自行传入)
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple:
        """单张图片推理,返回 (类别标签, 置信度)

        Args:
            image: PIL Image

        Returns:
            (label, confidence): label 为 class_names 中的名称或数字索引
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred = prob.max(dim=1)

        label = (
            self.class_names[pred.item()]
            if self.class_names is not None
            else pred.item()
        )
        return label, conf.item()

    @torch.no_grad()
    def predict_topk(self, image: Image.Image, k: int = 5) -> list[tuple]:
        """单张图片 Top-K 推理

        Args:
            image: PIL Image
            k: 返回前 k 个预测

        Returns:
            [(label, confidence), ...],按置信度降序排列
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        prob = torch.softmax(output, dim=1)

        topk_conf, topk_pred = prob.topk(k, dim=1)

        results = []
        for i in range(k):
            idx = topk_pred[0, i].item()
            conf = topk_conf[0, i].item()
            label = self.class_names[idx] if self.class_names is not None else idx
            results.append((label, conf))
        return results

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[tuple]:
        """批量推理

        Args:
            images: PIL Image 列表

        Returns:
            [(label, confidence), ...]
        """
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        outputs = self.model(batch)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)

        results = []
        for i in range(len(images)):
            label = (
                self.class_names[preds[i].item()]
                if self.class_names is not None
                else preds[i].item()
            )
            results.append((label, confs[i].item()))
        return results

    @torch.no_grad()
    def extract_features(
        self, image: Image.Image, layer_name: str | None = None
    ) -> torch.Tensor:
        """提取中间层特征

        Args:
            image: PIL Image
            layer_name: 目标层名称,None 则返回 logits 前一层(即 avgpool 输出)

        Returns:
            特征张量
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 如果未指定层名,直接用 AdaptiveAvgPool2d 前的特征
        if layer_name is None:
            # 通过 hook 捕获 features 层输出(大部分模型的倒数第二层)
            features = None

            def hook(module, input, output):
                nonlocal features
                features = output

            # 尝试在 avgpool 或 features 层注册 hook
            target = None
            for name, module in self.model.named_modules():
                if name in ("avgpool", "features"):
                    target = module
                    break

            if target is not None:
                handle = target.register_forward_hook(hook)
                _ = self.model(tensor)
                handle.remove()
                return features.flatten(1)

            # 回退:直接跑完整前向
            return self.model(tensor)

        # 指定层名时注册 hook
        features = None

        def hook(module, input, output):
            nonlocal features
            features = output

        target = dict(self.model.named_modules())[layer_name]
        handle = target.register_forward_hook(hook)
        _ = self.model(tensor)
        handle.remove()
        return features
