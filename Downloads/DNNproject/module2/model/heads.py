import torch.nn as nn


class LinearHead(nn.Module):
    """Head Variant A: single linear layer + softmax.

    Parameters added: 512 × 10 + 10 = 5,130
    """

    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        return self.softmax(self.fc(z))


class MLPHead(nn.Module):
    """Head Variant B: MLP with BN + Dropout for extra capacity.

    Parameters added: (512×256 + 256) + (256×10 + 10) = 133,898
    """

    def __init__(self, feature_dim=512, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        return self.softmax(self.net(z))
