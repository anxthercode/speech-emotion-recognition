"""Compact CNN producing six logits from [batch, 1, 64, 397] Log-Mel."""
from torch import nn


class EmotionCNN(nn.Module):
    def __init__(self, dropout=.2):
        super().__init__()
        layers = []
        previous = 1
        for channels in (32, 64, 128):
            layers.extend([nn.Conv2d(previous, channels, 3, padding=1), nn.BatchNorm2d(channels),
                           nn.ReLU(), nn.MaxPool2d(2)])
            previous = channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 6))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
