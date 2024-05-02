import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision.models as models
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

from .Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.conv_block1 = Conv_block(3, 64)
        # self.conv_block1 = ResidualConv(3, 64)
        # self.conv_block2 = ResidualConv(64, 128)
        # self.conv_block3 = ResidualConv(128, 256)
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)

        # self.conv_block1 = Conv_block(3, 32)


        self.l1 = nn.Linear(in_features=2048, out_features=2, bias=False)
        # self.l1 = nn.Linear(in_features=49, out_features=64, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        # self.l2 = nn.Linear(in_features=1024, out_features=2, bias=False)

        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.av1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.av2 = nn.AdaptiveAvgPool2d((1, 1))
        # self.av1 = nn.AvgPool2d(ih // 32, 1)


    def forward(self, x):
        x = self.resnet.conv1(x)  # torch.Size([16, 64, 192, 192])
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # torch.Size([16, 64, 96, 96])

        # ---- low-level features ----
        x = self.resnet.layer1(x)  # g2 # torch.Size([16, 256, 96, 96])
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.av1(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # x2 = self.conv_block2(images)
        # x2 = self.av2(x2)
        # x2 = x2.view(batch_size, -1)
        # x3 = torch.cat([x1, x2], dim=1)

        # x = self.conv_block3(x)
        # x = self.maxpool3(x)
        # #
        # x = self.conv_block4(x)
        # x = self.maxpool4(x)

        # x = self.av1(x)
        # batch_size = x.shape[0]
        # x = x.view(batch_size, -1)
        x = self.l1(x)
        # x4 = self.relu(x4)
        # x4 = self.l2(x4)
        output = x
        return output


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.cbr2_1 = BasicConv2drelu(in_channels, out_channels, 3, padding=1)
        self.cbr2_2 = BasicConv2drelu(out_channels, out_channels, 3, padding=1)
        self.se = Squeeze_Excite_block(out_channels)

    def forward(self, x):
        x = self.cbr2_1(x)
        x = self.cbr2_2(x)
        x = self.se(x)

        return x

class Squeeze_Excite_block(nn.Module):
    def __init__(self, out_channels):
        super(Squeeze_Excite_block, self).__init__()

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1, stride=1,
                               padding=0, dilation=1, bias=False)
        self.av1 = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(in_features=out_channels, out_features=out_channels // 8, bias=False)
        self.relu_a = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(in_features=out_channels // 8, out_features=out_channels, bias=False)
        self.sigmoid1 = nn.Sigmoid()

        # nn.init.xavier_uniform_(self.l2.weight.data)

        torch.nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.l2.weight, nonlinearity='sigmoid')

    def forward(self, x):
        batch_size = x.shape[0]
        # x1 = self.conv1(x)
        se = self.av1(x)
        se = se.reshape(batch_size, -1)
        se = self.l1(se)
        se = self.relu_a(se)
        se = self.l2(se)
        se = self.sigmoid1(se)
        se = se.reshape(batch_size, -1, 1, 1)
        x = torch.mul(x, se)
        return x



class BasicConv2drelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2drelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.bn = nn.LayerNorm(out_planes, elementwise_affine=False)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = FReLU(out_planes)

        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.bn.weight, 1)
        torch.nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv2d_bn(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv2d_bn, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(ResidualConv, self).__init__()

        self.cbri = BasicConv2drelu(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.cbrii = BasicConv2drelu(output_dim, output_dim, kernel_size=3, padding=1)
        self.conv_skip = Conv2d_bn(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
        self.se = Squeeze_Excite_block(output_dim)


    def forward(self, x):
        x1 = self.cbri(x)
        x1 = self.cbrii(x1)
        x2 = self.conv_skip(x)
        x3 = x1 + x2
        x3 = self.se(x3)

        return x3