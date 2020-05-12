import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from .modules import *
from .activations import get_activ


class BasicBlock(nn.Module):

    def __init__(self, n_channels, conv_weights, stride=1):
        super().__init__()

        #residual function
        self.w = conv_weights
        self.s = stride
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.bn2 = nn.BatchNorm2d(n_channels)

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        if stride != 1 :
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(stride),
                nn.BatchNorm2d(n_channels)
            )
        
    def forward(self, x):
        res = F.conv2d(x, self.w, stride=self.s, padding=1)
        res = self.bn1(res)
        res = F.relu(res)
        res = F.conv2d(x, self.w, stride=self.s, padding=1)
        res = self.bn2(res)
        return F.relu(res + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, input_shape, n_classes, n_filters, num_block):
        super().__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        
        self.conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False).weight

        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True))

        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(64, num_block[0], 1)
        self.conv3_x = self._make_layer(64, num_block[1], 2)
        self.conv4_x = self._make_layer(64, num_block[2], 2)
        self.conv5_x = self._make_layer(64, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_filters, n_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())
        self.pool_strategy = None

    def _make_layer(self, channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append( BasicBlock(self.n_filters, self.conv, stride)) 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))

        output = F.conv2d(x, self.conv, padding=1)
        output = self.bn1(output)
        
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output 

def resnet18(input_shape, n_classes, n_filters):
    return ResNet(input_shape, n_classes, n_filters, [2, 2, 2, 2])