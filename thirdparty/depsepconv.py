from torch import nn


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))
