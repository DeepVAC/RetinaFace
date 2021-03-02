import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from deepvac.syszux_modules import Conv2dBNPReLU, BottleneckIR, DepthWiseConv2d, initWeightsKaiming
from torch.nn import Module, init, Parameter
from typing import Any


class Resnet50IR(nn.Module):
    def __init__(self, embbeding_size):
        super(Resnet50IR, self).__init__()
        self.auditConfig()
        self.inplanes = 64

        self.conv1 = Conv2dBNPReLU(3, self.inplanes, 3, 1, 1)
        layers = []
        for outp, layer_num, stride in self.cfgs:
            layers.append(self.block(self.inplanes, outp, stride))
            self.inplanes = outp
            for _ in range(1, layer_num):
                layers.append(self.block(self.inplanes, outp, 1))

        self.layer = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm2d(512)
        self.dp1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512 * 7 * 7, embbeding_size)
        self.bn2 = nn.BatchNorm1d(embbeding_size, affine=False)

        initWeightsKaiming(self)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        x = self.bn1(x)
        x = x.reshape(x.size(0), -1)
        x = self.dp1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        return F.normalize(x)

    def auditConfig(self):
        self.block = BottleneckIR
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  2],
            [128,  4,  2],
            [256,  14, 2],
            [512,  3,  2]
        ] 


class ResNet100IR(Resnet50IR):
    def __init__(self,class_num: int = 1000):
        super(ResNet100IR, self).__init__(class_num)

    def auditConfig(self):
        self.block = BottleneckIR
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  2],
            [128,  13,  2],
            [256,  30,  2],
            [512,  3,  2]
        ]


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.auditConfig()
        self.inplanes = 64
        self.conv1 = Conv2dBNPReLU(3, self.inplanes, 3, 2, 1)
        layers =[]

        for outp, layer_num, stride, groups, residual in self.cfgs:
            for _ in range(layer_num):
                layers.append(self.block(self.inplanes, outp, 3, stride, 1, groups, residual))
            self.inplanes = outp
        self.layer = nn.Sequential(*layers)
        self.conv2 = Conv2dBNPReLU(self.inplanes, 512, 1, 1, 0)
        self.conv3 = nn.Conv2d(512, 512, 3, groups=512, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512*4*4, embedding_size, bias=False)
        # self.bn2 = nn.BatchNorm1d(embedding_size)
        initWeightsKaiming(self)


    def auditConfig(self):
        self.block = DepthWiseConv2d
        self.cfgs = [
            # outp, block_num, s, groups, residual
            # todo .maybe need 4 list:
            [64,   3,  1, 64, True],
            [64,   1,  2, 128, False],
            [64,   9,  1, 128, True],
            [128,  1,  2, 256, False],
            [128,  22, 1, 256, True],
            [128,  1,  2, 512, False],
            [128,  7,  1, 256, True]
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = self.bn2(x)
        return F.normalize(x)


class MobileFaceNetDDP(MobileFaceNet):
    def __init__(self, embedding_size):
        super(MobileFaceNetDDP, self).__init__(embedding_size)
        self.head = None

    def forward(self,x, label=None):
        x = super(MobileFaceNetDDP, self).forward(x)
        if label is None:
            return x
        return self.head(x, label)

    def extract(self,x):
        x = super(MobileFaceNetDDP, self).forward(x)
        return x

class Resnet50IRDDP(Resnet50IR):
    def __init__(self, embedding_size):
        super(Resnet50IRDDP, self).__init__(embedding_size)
        self.head = None

    def forward(self,x, label=None):
        x = super(Resnet50IRDDP, self).forward(x)
        if label is None:
            return x
        return self.head(x, label)

    def extract(self,x):
        x = super(Resnet50IRDDP, self).forward(x)
        return x
    

if __name__ == "__main__":
    model = Backbone(50, 0.4, 512)
    input = torch.ones((10,3,112,112))
    model(input)
