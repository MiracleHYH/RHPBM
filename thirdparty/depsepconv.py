from torch import nn


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.dep_sep_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2, stride=2, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        y = self.dep_sep_conv(x)
        return y
