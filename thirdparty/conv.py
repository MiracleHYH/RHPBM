from torch import nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicConvTranspose2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0):
        super(BasicConvTranspose2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, output_padding,
                                                 bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv_transpose(x)))
