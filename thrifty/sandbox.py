"""
/!\ Experimental stuff below

resnet in pytorch :
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1

https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .modules import MBConv
from .activations import get_activ

## ----------------- 1/ Resnets -------------------------------    
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
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
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output 

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

## ------ 2/ Factorized Resnet ---------------------

class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.stride = stride
        self.inc = in_channels
        self.outc = out_channels

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, conv1, conv2, sc_conv):

        # residual function
        res = F.conv2d(x, conv1[:self.outc, :self.inc, ...], stride=self.stride, padding=1)
        res = self.bn1(res)
        res = F.relu(res)
        
        res = F.conv2d(res, conv2[:self.outc, :self.outc, ...], padding=1)
        res = self.bn2(res)

        # shortcut
        if self.stride != 1 or self.inc != self.outc :
            sc = F.conv2d(x, sc_conv[:self.outc, :self.inc, ...], stride=self.stride)
            sc = self.bn3(sc)
        else:
            sc = x

        return F.relu(res + sc)

class FactorizedResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.embed = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Parameter(nn.Conv2d(256, 256, 3).weight)
        self.conv2 = nn.Parameter(nn.Conv2d(256, 256, 3).weight)
        self.sc_conv = nn.Parameter(nn.Conv2d(256, 256, 1).weight)
        
        self.bn1 = nn.BatchNorm2d(64)
        
        # resnet18 : [2, 2, 2, 2]
        self.blocks = nn.ModuleList([
            BasicBlock(64, 64, 2),
            BasicBlock(64, 64, 1),

            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1),

            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1),

            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 1)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        #x = F.conv2d(x, self.conv[:64, :3, ...], padding=1)
        x = self.embed(x)
        x = F.relu(self.bn1(x))
        for blck in self.blocks:
            x = blck(x, self.conv1, self.conv2, self.sc_conv)
        output = self.avg_pool(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output 

def factorized_resnet18(nb_classes):
    return FactorizedResNet(BasicBlock, nb_classes)

## ----------------- 3/ Unfactorized ThriftyNets -------------------------------

class OneConv(nn.Module):
    def __init__(self, n_filters, activ="relu", conv_mode="classic", bias=False):
        super(OneConv, self).__init__()
        self.n_filters = n_filters
        self.conv_mode = conv_mode
        self.bias = bias

        if self.conv_mode=="classic":
            self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        elif self.conv_mode=="mb1":
            self.Lconv = MBConv(n_filters, n_filters, bias=self.bias)
        elif self.conv_mode=="mb2":
            self.Lconv = MBConv(n_filters, n_filters//2, bias=self.bias)
        elif self.conv_mode=="mb4":
            self.Lconv = MBConv(n_filters, n_filters//4, bias=self.bias)
        self.activ = get_activ(activ)
        self.LBN = nn.BatchNorm2d(self.n_filters)

        self.alphas = torch.Tensor([0.1, 0.9])
        self.alphas = nn.Parameter(self.alphas)

    def forward(self, x):
        x2 = self.activ(self.Lconv(x))
        out = self.alphas[0] * x2 + self.alphas[1] * x
        return self.LBN(out)

class UnfactorThriftyNet(nn.Module):
    """
    A ThriftyNet where the convolution is no longer factorized
    """
    def __init__(self, input_shape, n_classes, n_filters, n_iter, pool_strategy, activ="relu", conv_mode="classic", bias=False):
        super(UnfactorThriftyNet, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_iter = n_iter

        self.pool_strategy = [False]*self.n_iter
        assert isinstance(pool_strategy, list) or isinstance(pool_strategy, tuple)
        if len(pool_strategy)==1:
            self.n_pool = 0
            freq = pool_strategy[0]
            for i in range(self.n_iter):
                if (i%freq == freq-1):
                    self.pool_strategy[i] = True
                    self.n_pool +=1
        else:
            self.n_pool = len(pool_strategy)
            for x in pool_strategy:
                self.pool_strategy[x] = True
       
        self.LOutput = nn.Linear(n_filters, n_classes)
        self.LConv = nn.ModuleList([OneConv(n_filters, activ, conv_mode, bias) for x in range(n_iter)])
        self.n_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        for t in range(self.n_iter):
            x = self.LConv[t](x)
            if self.pool_strategy[t]:
                x = F.max_pool2d(x, 2)
        out = F.adaptive_max_pool2d(x, (1,1))[:,:,0,0]
        return self.LOutput(out)

