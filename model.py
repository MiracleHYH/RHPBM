import torch
from torch import nn

from thirdparty import SE, Swish, DepthwiseSeparableConvolution


class EncoderCell(nn.Module):
    def __int__(self, channels):
        super(EncoderCell, self).__init__()
        self.process = nn.Sequential(
            nn.BatchNorm2d(channels),
            Swish(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(channels),
            Swish(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1),
            SE(channels)
        )

    def forward(self, x):
        y = self.process(x) + x
        return y


class DecoderCell(nn.Module):
    def __init__(self, channels, ext_channels):
        super(DecoderCell, self).__init__()
        self.process = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(in_channels=channels, out_channels=ext_channels, kernel_size=1),
            nn.BatchNorm2d(ext_channels),
            Swish(inplace=True),
            DepthwiseSeparableConvolution(in_channels=ext_channels, out_channels=ext_channels),
            nn.BatchNorm2d(ext_channels),
            Swish(inplace=True),
            nn.Conv2d(in_channels=ext_channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            SE(channels)
        )

    def forward(self, x):
        y = self.process(x) + x
        return y


class RHPBM(nn.Module):
    def __init__(self, height, width, d):
        super(RHPBM, self).__init__()

        self.height = height
        self.width = width
        self.feature_h = height
        self.feature_w = width

        for i in range(3):
            self.feature_h = (self.feature_h - 1) // 2 + 1
            self.feature_w = (self.feature_w - 1) // 2 + 1

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            self.relu,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_h * self.feature_w * 128, 2400),
            nn.Dropout(0.3),
            self.relu
        )

        self.mean_z = nn.Linear(2400, d)
        self.logvar_z = nn.Linear(2400, d)

        self.dfc = nn.Sequential(
            nn.Linear(d, 2400),
            nn.Linear(2400, self.feature_h * self.feature_w * 128),
            nn.Dropout(0.3),
            self.relu
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            self.relu,
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=2, output_padding=1),
            self.sigmoid
        )

    @staticmethod
    def reparam(mean_z, logvar_z):
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean_z)

    def forward(self, x):
        z = self.encode(x)
        z = self.fc(z.view(-1, self.feature_h * self.feature_w * 128))
        mean_z = self.mean_z(z)
        logvar_z = self.logvar_z(z)
        _z = RHPBM.reparam(mean_z, logvar_z)
        _x = self.dfc(_z)
        _x = self.decode(_x.view(-1, 128, self.feature_h, self.feature_w))
        return _x, mean_z, logvar_z
