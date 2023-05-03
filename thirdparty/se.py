import torch
from torch import nn


class SE(nn.Module):
    def __init__(self, channels):
        super(SE, self).__init__()
        num_hidden = max(channels // 16, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channels, num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y
