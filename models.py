import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

import numpy as np
from modules import *
from functools import partial

"""

Parameters
----------
args : parsed command line argument dict

metadata : dict
    The metadatas of the dataset to consider, especially the size of each inputs and the number of classes
"""
def get_model(args, metadata):
    model_name = args.model.lower()    
    if model_name=="thrifty":
        return ThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, 
                        args.iter, args.pool, args.activ, args.conv_mode, args.out_mode, args.bias)

    elif model_name in ["res_thrifty", "resthrifty"]:
        return ResThriftyNet(metadata["input_shape"], metadata["n_classes"], n_filters=args.filters, 
                            n_iter=args.iter, n_history=args.history, pool_strategy=args.pool, 
                            activ=args.activ, conv_mode=args.conv_mode, out_mode=args.out_mode, bias=args.bias)
    
    elif model_name in ["block_thrifty", "blockthrifty"]:
        return ThriftyNet_3State(metadata["input_shape"], metadata["n_classes"], 
                                 n_history=args.history, conv_mode=args.conv_mode,
                                 activ=args.activ, bias=args.bias)
    elif model_name in ["embedded_thrifty", "embeddedthrifty"]:
        return EmbeddedThriftyNet(args.filters, args.iter)

    else:
        raise Exception("Model type was not recognized")

class ThriftyNet(nn.Module):
    """
    Just a ResThriftyNet with history = 1
    """
    def __init__(self, input_shape, n_classes, n_filters, n_iter, pool_strategy, activ="relu", conv_mode="classic", out_mode="pool", bias=False):
        super(ThriftyNet, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.out_mode = out_mode
        self.Lblock = ThriftyBlock(n_filters, n_iter, 1, pool_strategy, conv_mode=conv_mode, activ=activ, bias=bias)
        
        if out_mode == "pool":
            out_size = n_filters
        elif out_mode == "flatten":
            out_size = np.prod(self.Lblock.out_shape(input_shape)) 
        self.LOutput = nn.Linear(out_size, n_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())

        @property
        def n_filters(self):
            return self.Lblock.n_filters

        @property
        def pool_strategy(self):
            return self.Lblock.pool_strategy

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        x = self.Lblock(x)
        if self.out_mode=="pool" and x.size()[-1]>1:
            out = F.adaptive_max_pool2d(x, (1,1))[:,:,0,0]
        elif self.out_mode=="flatten":
            out = x.view(x.size()[0], -1)
        else:
            out = x[:,:,0,0]
        return self.LOutput(out)


class ResThriftyNet(nn.Module):
    """
    Residual Thrifty Network
    """
    def __init__(self, input_shape, n_classes, n_filters, n_iter, n_history, pool_strategy, activ="relu", conv_mode="classic", out_mode="pool", bias=False):
        super(ResThriftyNet, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.Lblock = ThriftyBlock(n_filters, n_iter, n_history, pool_strategy, conv_mode=conv_mode, activ=activ, bias=bias)
        self.out_mode = out_mode

        if out_mode == "pool":
            out_size = n_filters
        elif out_mode == "flatten":
            out_size = np.prod(self.Lblock.out_shape(input_shape)) 
        self.LOutput = nn.Linear(out_size, n_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())

    @property
    def n_filters(self):
        return self.Lblock.n_filters

    @property
    def pool_strategy(self):
        return self.Lblock.pool_strategy

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        x = self.Lblock(x)
        if self.out_mode=="pool" and x.size()[-1]>1:
            out = F.adaptive_max_pool2d(x, (1,1))[:,:,0,0]
        elif self.out_mode=="flatten":
            out = x.view(x.size()[0], -1)
        else:
            out = x[:,:,0,0]
        return self.LOutput(out)


class ThriftyNet_3State(nn.Module):
    """
    Three ThriftNets stacked on top of each other
    """
    def __init__(self, input_shape, n_classes, n_history, conv_mode="classic", activ="relu", bias=False):
        super(ThriftyNet_3State, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activ = activ
        self.conv_mode = conv_mode
        self.bias = bias
        self.n_history = n_history

        self.n_filters = [200, 500]
        self.n_iters = [30, 40]
        self.pool_strategy = None

        self.block1 = ThriftyBlock(self.n_filters[0], self.n_iters[0], self.n_history, [14,29], conv_mode=conv_mode, activ=activ, bias=bias)
        self.block2 = ThriftyBlock(self.n_filters[1], self.n_iters[1], self.n_history, [14,29], conv_mode=conv_mode, activ=activ, bias=bias)
        #self.block3 = ThriftyBlock(self.n_filters[2], self.n_iters[2], self.n_history, [12, 24], conv_mode=conv_mode, activ=activ, bias=bias)

        self.LOutput = nn.Linear(4*self.n_filters[-1], self.n_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x0 = F.pad(x, (0, 0, 0, 0, 0, self.n_filters[0]-3))
        x1 = self.block1(x0)
        x1 = F.pad(x1, (0, 0, 0, 0, 0, self.n_filters[1] - self.n_filters[0]))
        x2 = self.block2(x1)
        #x2 = F.pad(x2, (0, 0, 0, 0, 0, self.n_filters[2] - self.n_filters[1]))
        #x3 = self.block3(x2)

        #out = F.adaptive_max_pool2d(x3, (1,1))[:,:,0,0]
        out = x2.view(x2.size()[0], -1)
        return self.LOutput(out)

class EmbeddedThriftyNet(nn.Module):
    """
    The first blocks of a Resnet, followed by a Thrifty block
    """

    def __init__(self, n_filters, n_iter, n_history, pool_strategy, activ="relu", conv_mode="classic", bias=False):
        super(EmbeddedThriftyNet, self).__init__()
        self.embed_shape = (128,28,28) # output shape of the embedder
        
        self.Lembed = ResNetEmbedder(resnet.BasicBlock, [3, 4])
        self.Lthrifty = ThriftyBlock(n_filters, n_iter, n_history, pool_strategy,conv_mode=conv_mode, activ=activ, bias=bias)
        self.LOutput = nn.Linear(n_filters, self.n_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())
        print("Embedding parameters: ", self.Lembed.n_parameters)
        print("Thrifty parameters: ", self.Lthrifty.n_parameters)
        print("Total parameters: ", self.n_parameters)

    def forward(self, x):        
        x = self.Lembed(x)
        x = F.pad(x, (2, 2, 2, 2, 0, self.n_filters - self.embed_shape[0]))
        x = self.Lthrifty(x) 
        out = F.adaptive_max_pool2d(x, (1,1))[:,:,0,0]
        return self.LOutput(out)
