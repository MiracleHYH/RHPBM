import torch
from torch import nn

from thirdparty import SE, Swish, DepthwiseSeparableConvolution


class EncoderCell(nn.Module):
    def __int__(self, in_channels, out_channels, stride):
        super(EncoderCell, self).__init__()
        self.swish = Swish()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.se = SE(out_channels)

        if stride > 1:
            self.rconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.conv1(self.swish(self.bn1(x)))
        y = self.conv2(self.swish(self.bn2(x)))
        y = self.se(y)
        if self.rconv:
            x = self.rconv(x)
        return y+x


class DecoderCell(nn.Module):
    def __init__(self, in_channels, out_channels, stride, ext):
        super(DecoderCell, self).__init__()
        ext_channels = in_channels * ext
        self.swish = Swish()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=ext_channels, kernel_size=1, stride=ext)
        self.bn2 = nn.BatchNorm2d(ext_channels)
        self.conv2 = DepthwiseSeparableConvolution(in_channels=ext_channels, out_channels=ext_channels)
        self.bn3 = nn.BatchNorm2d(ext_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels=ext_channels, out_channels=in_channels, kernel_size=1, stride=ext)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.conv4 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, output_padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.se = SE(out_channels)

        if stride > 1:
            self.rconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.bn1(x)
        y = self.swish(self.bn2(self.conv1(y)))
        y = self.swish(self.bn3(self.conv2(y)))
        y = self.swish(self.bn4(self.conv3(y)))
        y = self.bn5(self.conv4(y))
        y = self.se(y)
        if self.rconv:
            x = self.rconv(x)
        return y+x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, block_type):
        super(Block, self).__init__()
        if block_type == 'encoder_block':
            cell1 = EncoderCell(in_channels, out_channels, stride)
            cell2 = EncoderCell(out_channels, out_channels, 1)
        else:
            cell1 = DecoderCell(in_channels, out_channels, stride, 2)
            cell2 = DecoderCell(out_channels, out_channels, 1, 2)

    def forward(self, x):
        y = self.cell1(x)
        y = self.cell2(x)
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

        self.front_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.front_bn = nn.BatchNorm2d(32)
        self.enc_block1 = Block(in_channels=32, out_channels=64, stride=2, block_type='encoder_block')
        self.enc_block2 = Block(in_channels=64, out_channels=128, stride=2, block_type='encoder_block')
        self.fc = nn.Linear(self.feature_h * self.feature_w * 128, 2400)
        self.drop = nn.Dropout(0.3)
        self.mean_z = nn.Linear(2400, d)
        self.logvar_z = nn.Linear(2400, d)

        self.dfc1 = nn.Linear(d, 2400)
        self.dfc2 = nn.Linear(2400, self.feature_h * self.feature_w * 128)
        self.ddrop = nn.Dropout(0.3)
        self.dec_block1 = Block(in_channels=128, out_channels=64, stride=2, block_type='decoder_block')
        self.dec_block2 = Block(in_channels=64, out_channels=32, stride=2, block_type='decoder_block')
        self.end_conv = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, output_padding=1)

    def encode(self, x):
        z = self.relu(self.front_bn(self.front_conv(x)))
        z = self.enc_block1(z)
        z = self.enc_block2(z)
        z = self.relu(self.drop(self.fc(z.view(-1, self.feature_h * self.feature_w * 128))))
        return self.mean_z(z), self.logvar_z(z)

    def decode(self, z):
        _x = self.relu(self.dfc1(z))
        _x = self.relu(self.ddrop(self.dfc2(_x)))
        _x = self.dec_block1(_x.view(-1, 128, self.feature_h, self.feature_w))
        _x = self.dec_block2(_x)
        _x = self.sigmoid(self.end_conv(_x))
        return _x

    @staticmethod
    def reparam(mean_z, logvar_z):
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean_z)

    def forward(self, x):
        mean_z, logvar_z = self.encode(x)
        _z = RHPBM.reparam(mean_z, logvar_z)
        return self.decode(_z), mean_z, logvar_z
