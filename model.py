import torch
from torch import nn

from thirdparty import SE, Swish, DepthwiseSeparableConvolution


class EncoderCell(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(EncoderCell, self).__init__()
        self.swish = Swish()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.se = SE(out_channels)

        self.shortcut = None
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        y = self.conv1(self.swish(self.bn1(x)))
        y = self.conv2(self.swish(self.bn2(y)))
        y = self.se(y)
        if self.rconv:
            x = self.shortcut(x)
        return y + x


class DecoderCell(nn.Module):
    def __init__(self, in_channels, out_channels, stride, ext):
        super(DecoderCell, self).__init__()
        self.tag = stride > 1

        ext_channels = in_channels * ext
        self.swish = Swish()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=ext_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(ext_channels)
        self.conv2 = DepthwiseSeparableConvolution(in_channels=ext_channels, out_channels=ext_channels)
        self.bn3 = nn.BatchNorm2d(ext_channels)
        self.conv3 = nn.Conv2d(in_channels=ext_channels, out_channels=in_channels, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(in_channels)
        if self.tag:
            self.conv4 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                            stride=stride, padding=1, output_padding=1)
            self.bn5 = nn.BatchNorm2d(out_channels)
        self.se = SE(out_channels)
        if self.tag:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   output_padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # print("--------------------------")
        # print('input:', x.shape)
        y = self.bn1(x)
        y = self.swish(self.bn2(self.conv1(y)))
        # print('conv 1x1:', y.shape)
        y = self.swish(self.bn3(self.conv2(y)))
        # print('dep. sep. conv:', y.shape)
        y = self.swish(self.bn4(self.conv3(y)))
        # print('dconv 1x1:', y.shape)
        if self.tag:
            y = self.bn5(self.conv4(y))
        # print('dconv 3x3:', y.shape)
        y = self.se(y)
        if self.tag:
            x = self.shortcut(x)
        # print('final x,y:', x.shape, y.shape)
        return y + x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, block_type):
        super(Block, self).__init__()
        if block_type == 'encoder_block':
            self.cell1 = EncoderCell(in_channels, out_channels, stride)
            self.cell2 = EncoderCell(out_channels, out_channels, 1)
        else:
            self.cell1 = DecoderCell(in_channels, out_channels, stride, 2)
            self.cell2 = DecoderCell(out_channels, out_channels, 1, 2)

    def forward(self, x):
        y = self.cell1(x)
        y = self.cell2(y)
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

        self.front_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.front_bn = nn.BatchNorm2d(32)
        self.enc_block2 = Block(in_channels=32, out_channels=64, stride=2, block_type='encoder_block')
        self.enc_block3 = Block(in_channels=64, out_channels=128, stride=2, block_type='encoder_block')
        self.enc_block4 = Block(in_channels=128, out_channels=256, stride=2, block_type='encoder_block')
        self.fc = nn.Linear(self.feature_h * self.feature_w * 256, 1024)
        self.drop = nn.Dropout(0.3)
        self.mean_z = nn.Linear(1024, d)
        self.logvar_z = nn.Linear(1024, d)

        self.dfc1 = nn.Linear(d, 1024)
        self.dfc2 = nn.Linear(1024, self.feature_h * self.feature_w * 1024)
        self.ddrop = nn.Dropout(0.3)
        self.dec_block1 = Block(in_channels=256, out_channels=128, stride=2, block_type='decoder_block')
        self.dec_block2 = Block(in_channels=128, out_channels=64, stride=2, block_type='decoder_block')
        self.dec_block3 = Block(in_channels=64, out_channels=32, stride=2, block_type='decoder_block')
        self.end_conv = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
        self.end_bn = nn.BatchNorm2d(32)

    def encode(self, x):
        # print('-------------encoder-------------')
        # print("[Input]: ", x.shape)
        z = self.relu(self.front_bn(self.front_conv(x)))
        # print("[Front Layer]: ", z.shape)
        z = self.enc_block1(z)
        # print("[Encode Layer 1]: ", z.shape)
        z = self.enc_block2(z)
        # print("[Encode Layer 2]: ", z.shape)
        z = self.enc_block3(z)
        # print("[Encode Layer 5]: ", z.shape, z.view(-1, self.feature_h * self.feature_w * 256).shape)
        z = self.relu(self.drop(self.fc(z.view(-1, self.feature_h * self.feature_w * 256))))
        # print("[FC Layer]: ", z.shape)
        return self.mean_z(z), self.logvar_z(z)

    def decode(self, z):
        # print('-------------decoder-------------')
        # print("[Input]: ", z.shape)
        _x = self.relu(self.dfc1(z))
        # print("[FC Layer1]: ", _x.shape)
        _x = self.relu(self.ddrop(self.dfc2(_x)))
        # print("[FC Layer2]: ", _x.shape, _x.view(-1, 256, self.feature_h, self.feature_w).shape)
        _x = self.dec_block1(_x.view(-1, 256, self.feature_h, self.feature_w))
        # print("[Decode Layer1]: ", _x.shape)
        _x = self.dec_block2(_x)
        # print("[Decode Layer2]: ", _x.shape)
        _x = self.dec_block3(_x)
        # print("[Decode Layer3]: ", _x.shape)
        _x = self.sigmoid(self.end_conv(self.end_bn(_x)))
        # print("[Output]: ", _x.shape)
        return _x

    @staticmethod
    def reparam(mean_z, logvar_z):
        std = logvar_z.mul(0.5).exp()
        eps = torch.randn_like(std).to('cuda:0')
        return eps.mul(std).add_(mean_z)

    def forward(self, x):
        mean_z, logvar_z = self.encode(x)
        _z = RHPBM.reparam(mean_z, logvar_z)
        return self.decode(_z), mean_z, logvar_z
