"""
ResNet-18 backbone.

Reference: "Deep Residual Learning for Image Recognition" (He et al., 2015).
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Two 3x3 conv layers with a skip connection.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 feature extractor. Returns feature maps at stride 16 and stride 32.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat16 = self.layer3(x)
        feat32 = self.layer4(feat16)
        return {"feat16": feat16, "feat32": feat32}

    def load_pretrained(self):
        """
        Load ImageNet-1k weights from torchvision. We verify parameter names
        match ours; torchvision's resnet18 uses the same layer1..layer4 layout.
        """
        from torchvision.models import resnet18, ResNet18_Weights
        pretrained = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        missing, unexpected = self.load_state_dict(pretrained, strict=False)
        print(f"Loaded pretrained ResNet-18: {len(pretrained)} tensors")
        if missing:
            print(f"  Missing: {missing}")
        if unexpected:
            print(f"  Unexpected: {unexpected}")
