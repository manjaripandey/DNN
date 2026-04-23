import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet18CIFAR(nn.Module):
    """ResNet-18 backbone adapted for 32×32 CIFAR images.

    Uses a 3×3 conv stem with stride=1 and no max-pool, preserving
    spatial resolution through the stem (standard CIFAR adaptation).
    Outputs a 512-dimensional feature vector after global average pooling.
    No classification head is included — attach LinearHead or MLPHead separately.
    """

    def __init__(self):
        super().__init__()
        # CIFAR-adapted stem: no spatial downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, stride=1)   # → (B, 64, 32, 32)
        self.layer2 = self._make_layer(64, 128, stride=2)  # → (B, 128, 16, 16)
        self.layer3 = self._make_layer(128, 256, stride=2) # → (B, 256, 8, 8)
        self.layer4 = self._make_layer(256, 512, stride=2) # → (B, 512, 4, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)                 # → (B, 512, 1, 1)
        self.feature_dim = 512

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)  # (B, 512)
