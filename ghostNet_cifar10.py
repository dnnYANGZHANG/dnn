import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import argparse
from torchsummary import summary

from ghost import *

class LastConvLayer(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU, groups=8):
        super(LastConvLayer, self).__init__()
        self.groups = groups
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False, groups=self.groups)
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = channel_shuffle(x, self.groups)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class GhostNet(nn.Module):
    def __init__(self):
        super(GhostNet, self).__init__()
        # First Layer
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = 8

        # Second Layer
        self.cfgs = [
            [[3, 24, 16, 0, 1, 4, 0]],
            [[3, 48, 32, 0, 1, 4, 0]],
            [[3, 64, 48, 0, 1, 4, 0]],
            [[3, 72, 64, 0, 1, 4, 0]],
            [[3, 120, 72, 0, 1, 4, 0]],
            [[3, 240, 80, 0, 1, 8, 0]],
            [[3, 240, 80, 0, 1, 8, 1],
             [3, 240, 80, 0, 1, 8, 1],
             [3, 240, 112, 0, 1, 8, 0]],
            [[3, 480, 160, 0, 1, 16, 0],
             [3, 480, 160, 0, 1, 16, 1],
             [3, 480, 160, 0, 1, 16, 1]]
        ]
        stages = []
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s, g, p in cfg:
                output_channel = c
                hidden_channel = exp_size
                layers.append(GhostBottleneckV022(input_channel, hidden_channel, output_channel, group=g, dw_kernel_size=k,
                                              stride=s, padding=p, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        stages.append(nn.Sequential(LastConvLayer(input_channel, 480, 1, groups=16)))
        self.blocks = nn.Sequential(*stages)

        # Last Several Layers
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        # self.global_pool = nn.AvgPool2d(8)
        # self.conv_last = nn.Conv2d(480, 640, 1, 1, 0, bias=True)
        # self.act_last = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(7680, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        # x = self.conv_last(x)
        # x = self.act_last(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class GhostNetV022(GhostNet):
    def __init__(self):
        super().__init__()
        input_channel = 8
        self.cfgs = [
            [[3, 24, 16, 0, 1, 4, 0]],
            [[3, 48, 32, 0.25, 1, 4, 0]],
            [[3, 64, 48, 0, 1, 4, 0]],
            [[3, 72, 64, 0.25, 1, 4, 0]],
            [[3, 120, 72, 0, 1, 4, 0]],
            [[3, 240, 80, 0, 1, 8, 0]],
            [[3, 240, 80, 0.25, 1, 8, 1],
             [3, 240, 80, 0, 1, 8, 1],
             [3, 240, 112, 0, 1, 8, 0]],
            [[3, 480, 160, 0.25, 1, 16, 0],
             [3, 480, 160, 0, 1, 16, 1],
             [3, 480, 160, 0, 1, 16, 1]]
        ]
        stages = []
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s, g, p in cfg:
                output_channel = c
                hidden_channel = exp_size
                layers.append(GhostBottleneckV022(input_channel, hidden_channel, output_channel, group=g, dw_kernel_size=k,
                                                  stride=s, padding=p, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        stages.append(nn.Sequential(LastConvLayer(input_channel, 480, 1, groups=16)))
        self.blocks = nn.Sequential(*stages)

        # Last Several Layers
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        # self.global_pool = nn.AvgPool2d(8)
        # self.conv_last = nn.Conv2d(480, 640, 1, 1, 0, bias=True)
        # self.act_last = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(7680, 10)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GhostNetV022().to(device)
    summary(model, (3, 28, 28))






