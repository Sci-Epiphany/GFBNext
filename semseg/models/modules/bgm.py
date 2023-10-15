
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BGM(nn.Module):
    def __init__(self, dims):  # input dims: embed_dims
        super(BGM, self).__init__()

        self.reduce = Conv1x1(dims[0], 64)
        self.block = nn.Sequential(
            ConvBNR(64, 32, 3),
            ConvBNR(32, 16, 3),
            nn.Conv2d(16, 4, 1)
        )
        self.pred = nn.Conv2d(4, 1, 1)

    def forward(self, x):

        x = self.reduce(x)
        x = self.block(x)
        out = self.pred(x)

        return out

