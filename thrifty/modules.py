import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .activations import get_activ
from .resnet_models import *

class MBConv(nn.Module):
    def __init__(self, in_maps, compress, bias=False):
        super(MBConv, self).__init__()
        self.conv1 = nn.Conv2d(in_maps, compress, kernel_size = 3, padding = 1, bias = bias, groups = compress)
        self.conv2 = nn.Conv2d(compress, in_maps, kernel_size = 1, padding = 0, bias = bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))

def Conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class ThriftyBlock(nn.Module):
    def __init__(self, n_filters, n_iter, n_history, pool_strategy, conv_mode="classic", activ="relu", bias=False):
        super(ThriftyBlock, self).__init__()
        self.n_filters = n_filters
        self.n_iter = n_iter
        self.n_history = n_history
        self.activ = activ
        self.conv_mode = conv_mode
        self.bias = bias
        
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

        self.Lactiv = get_activ(activ)
        self.Lnormalization = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])

        if self.conv_mode=="classic":
            self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        elif self.conv_mode=="mb1":
            self.Lconv = MBConv(n_filters, n_filters, bias=self.bias)
        elif self.conv_mode=="mb2":
            self.Lconv = MBConv(n_filters, n_filters//2, bias=self.bias)
        elif self.conv_mode=="mb4":
            self.Lconv = MBConv(n_filters, n_filters//4, bias=self.bias)
        self.activ = get_activ(activ)
        
        self.alpha = torch.zeros((n_iter, n_history+1))
        for t in range(n_iter):
            self.alpha[t,0] = 0.1
            self.alpha[t,1] = 0.9
        self.alpha = nn.Parameter(self.alpha)

        self.n_parameters = sum(p.numel() for p in self.parameters())

    def out_shape(self,input_shape):
        """
        Computes the output tensor's shape, given a specific input shape
        """
        n_filters, x, y = input_shape
        ds = 2**(self.n_pool)
        return (self.n_filters, x // ds, y // ds)


    def forward(self, x):
        hist = [None for _ in range(self.n_history-1)] + [x]

        for t in range(self.n_iter):
            a = self.Lconv(hist[-1])
            a = self.Lactiv(a)
            a = self.alpha[t,0] * a
            for i, x in enumerate(hist):
                if x is not None:
                    a = a + self.alpha[t,i+1] * x

            a = self.Lnormalization[t](a)
            
            for i in range(1, self.n_history-1):
                hist[i] = hist[i+1]
            hist[self.n_history-1] = a

            if self.pool_strategy[t]:
                for i in range(len(hist)):
                    if hist[i] is not None:
                        hist[i] = F.max_pool2d(hist[i], 2)
        return hist[-1]


class ResNetEmbedder(nn.Module):

    def __init__(self, in_channels, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, self.in_channels, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 2*self.in_channels, num_block[1], 2)
        #self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        #self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())

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
        #output = self.conv4_x(output)
        #output = self.conv5_x(output)
        #output = self.avg_pool(output)
        #output = output.view(output.size(0), -1)
        #output = self.fc(output)
        return output