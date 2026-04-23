import torch.nn as nn
from .backbone import ResNet18CIFAR
from .heads import LinearHead, MLPHead, TemperatureHead


class FullModel(nn.Module):
    """Backbone + one prediction head. forward() returns a softmax probability vector.

    head_variant: 'A' = LinearHead, 'B' = MLPHead, 'C' = TemperatureHead
    """

    def __init__(self, head_variant='A'):
        super().__init__()
        self.backbone = ResNet18CIFAR()
        if head_variant == 'A':
            self.head = LinearHead(self.backbone.feature_dim, num_classes=10)
        elif head_variant == 'B':
            self.head = MLPHead(self.backbone.feature_dim, hidden=256, num_classes=10)
        elif head_variant == 'C':
            self.head = TemperatureHead(self.backbone.feature_dim, num_classes=10)
        else:
            raise ValueError(f"head_variant must be 'A', 'B', or 'C', got '{head_variant}'")

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)
