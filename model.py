import torch
from torch import nn

from thirdparty import BasicConv2d, BasicConvTranspose2d


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            BasicConv2d(in_planes=3, out_planes=32, kernel_size=3, stride=2, padding=0),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=0),
            BasicConv2d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
            # BasicConv2d(in_planes=64, out_planes=64, kernel_size=3, stride=2, padding=0)
            BasicConv2d(in_planes=64, out_planes=80, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=80, out_planes=192, kernel_size=3, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=256, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        y = self.stem(x)
        return y


class InceptionResnetA(nn.Module):
    def __init__(self, scale=0.1):
        super(InceptionResnetA, self).__init__()

        self.scale = scale

        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=32, kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=32, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=32, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        y = self.conv(y)
        return self.relu(y * self.scale + x)


class InceptionResnetB(nn.Module):
    def __init__(self, scale=0.1):
        super(InceptionResnetB, self).__init__()

        self.scale = scale

        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=128, kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=128, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=128, out_planes=128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=128, out_planes=128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.conv = nn.Conv2d(in_channels=256, out_channels=896, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        y = self.conv(y)
        return self.relu(y * self.scale + x)


class InceptionResnetC(nn.Module):
    def __init__(self, scale=0.1):
        super(InceptionResnetC, self).__init__()

        self.scale = scale

        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=1792, out_planes=192, kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=1792, out_planes=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv = nn.Conv2d(in_channels=384, out_channels=1792, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        y = self.conv(y)
        return self.relu(y * self.scale + x)


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2, padding=0)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=256, out_planes=192, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=192, out_planes=256, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        y = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return y


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2, padding=0)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=2, padding=0)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        y = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return y


class StemTranspose(nn.Module):
    def __init__(self):
        super(StemTranspose, self).__init__()
        self.stem_transpose = nn.Sequential(
            BasicConvTranspose2d(in_planes=256, out_planes=192, kernel_size=3, stride=2, padding=0, output_padding=0),
            BasicConvTranspose2d(in_planes=192, out_planes=80, kernel_size=3, stride=1, padding=0, output_padding=0),
            BasicConvTranspose2d(in_planes=80, out_planes=64, kernel_size=1, stride=1, padding=0, output_padding=0),
            BasicConvTranspose2d(in_planes=64, out_planes=64, kernel_size=3, stride=2, padding=0, output_padding=0),
            BasicConvTranspose2d(in_planes=64, out_planes=32, kernel_size=3, stride=1, padding=1, output_padding=0),
            BasicConvTranspose2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.Sigmoid()
            # BasicConvTranspose2d(in_planes=32, out_planes=3, kernel_size=3, stride=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        y = self.stem_transpose(x)
        return y


class RHPBM(nn.Module):
    def __init__(self, d):
        super(RHPBM, self).__init__()

        layers = list()
        layers.append(Stem())
        for i in range(5):
            layers.append(InceptionResnetA(scale=0.1))
        layers.append(ReductionA())
        for i in range(10):
            layers.append(InceptionResnetB(scale=0.1))
        layers.append(ReductionB())
        for i in range(5):
            layers.append(InceptionResnetC(scale=0.1))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Dropout(0.3))
        self.inception = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1792, out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.mu_z = nn.Linear(in_features=512, out_features=d)
        self.logvar_z = nn.Linear(in_features=512, out_features=d)

        self.dfc = nn.Sequential(
            nn.Linear(in_features=d, out_features=512),
            nn.Linear(in_features=512, out_features=1792),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(in_features=1792, out_features=1792*8*8, bias=False),
            nn.BatchNorm1d(1792*8*8),
            nn.ReLU()
        )

        self.recover = nn.Sequential(
            BasicConvTranspose2d(in_planes=1792, out_planes=896, kernel_size=3, stride=2, padding=0, output_padding=0),
            BasicConvTranspose2d(in_planes=896, out_planes=256, kernel_size=3, stride=2, padding=0, output_padding=0),
            StemTranspose()
        )

    def encode(self, x):
        z = self.inception(x)
        z = self.fc(z.view(-1, 1792))
        mu_z = self.mu_z(z)
        logvar_z = self.logvar_z(z)
        return mu_z, logvar_z

    def decode(self, z):
        _x = self.dfc(z)
        _x = self.recover(_x.view(-1, 1792, 8, 8))
        return _x

    @staticmethod
    def reparam(mu_z, logvar_z):
        std = logvar_z.mul(0.5).exp()
        eps = torch.randn_like(std).to('cuda:0')
        return eps.mul(std).add_(mu_z)

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        _z = RHPBM.reparam(mu_z, logvar_z)
        return self.decode(_z), mu_z, logvar_z
