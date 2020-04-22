import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return ThriftyNet(metadata["input_shape"], metadata["n_classes"], args.size, 
                        args.iter, args.pool, args.activ, args.conv_mode, args.bias)

    elif model_name in ["res_thrifty", "resthrifty"]:
        return ResThriftyNet(metadata["input_shape"], metadata["n_classes"], n_filters=args.size, 
                            n_iter=args.iter, n_history=args.history, pool_strategy=args.pool, 
                            activ=args.activ, conv_mode=args.conv_mode, bias=args.bias)
    
    elif model_name in ["block_thrifty", "blockthrifty"]:
        return ThriftyNet_3State(metadata["input_shape"], metadata["n_classes"], 
                                 n_history=args.history, conv_mode=args.conv_mode,
                                 activ=args.activ, bias=args.bias)

    else:
        raise Exception("Model type was not recognized")

class ThriftyNet(nn.Module):

    def __init__(self, input_shape, n_classes, n_filters, n_iter, pool_strategy, activ="relu", conv_mode="classic", bias=False):
        super(ThriftyNet, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_iter = n_iter
        self.activ = activ
        self.conv_mode = conv_mode
        self.bias = bias
        
        self.pool_strategy = [False]*self.n_iter
        assert isinstance(pool_strategy, list) or isinstance(pool_strategy, tuple)
        if len(pool_strategy)==1:
            freq = pool_strategy[0]
            for i in range(self.n_iter):
                if (i%freq == freq-1):
                    self.pool_strategy[i] = True
        else:
            for x in pool_strategy:
                self.pool_strategy[x] = True

        self.Lactiv = get_activ(activ)
        self.Lnormalization = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])

        if self.conv_mode=="classic":
            self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        elif self.conv_mode=="mb1":
            self.Lconv = MBConv(n_filters, n_filters)
        elif self.conv_mode=="mb2":
            self.Lconv = MBConv(n_filters, n_filters//2)
        elif self.conv_mode=="mb4":
            self.Lconv = MBConv(n_filters, n_filters//4)

        self.LOutput = nn.Linear(n_filters, n_classes)
        self.activ = get_activ(activ)

        self.n_parameters = sum(p.numel() for p in self.parameters())


    def forward(self, x, get_features=False):
        xcur = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        
        for t in range(self.n_iter):
            xnext = self.Lconv(xcur)
            xnext = self.alpha[t,0] * self.Lactiv(xnext) + self.alpha[t,1] * xcur
            xnext = self.Lnormalization[t](xnext)
            if self.pool_strategy[t]:
                xnext = F.max_pool2d(xnext, 2)
        
        if xnext.size()[-1]>1:
            out = F.adaptive_max_pool2d(xnext, (1,1))[:,:,0,0]
        else:
            out = xnext
            
        if get_features:
            return out, self.LOutput(out)
        else:
            return self.LOutput(out)

    def save(self, path):
        data = { "input_shape" : self.input_shape,
                 "conv_mode" : self.conv_mode,
                 "n_classes" : self.n_classes,
                 "n_filters" : self.n_filters,
                 "n_iter" : self.n_iter,
                 "bias" : self.bias,
                 "activ" : self.activ,
                 "pool_strategy" : self.pool_strategy,
                 "state_dict" : self.state_dict()}
        torch.save(data, path)


    @staticmethod
    def fromFile(path):
        map_loc = "cpu" if not torch.cuda.is_available() else None
        data = torch.load(path, map_location=map_loc)
        model = ThriftyNet(data["input_shape"], data["n_classes"], 
                            data["n_filters"], n_iter=data["iter"], 
                            pool_strategy=[0], activ=data["activ"], 
                            conv_mode = data["conv_mode"], bias=data["bias"])
        model.pool_strategy = data["pool_strategy"]
        model.load_state_dict(data["state_dict"])
        return model

class ResThriftyNet(ThriftyNet):

    def __init__(self, input_shape, n_classes, n_filters, n_iter, n_history, pool_strategy, activ="relu", conv_mode="classic", bias=False):
        ThriftyNet.__init__(self, input_shape, n_classes, n_filters, n_iter, pool_strategy, activ=activ, conv_mode=conv_mode, bias=bias)
        self.n_history = n_history
        
        self.alpha = torch.zeros((n_iter, n_history+1))
        for t in range(n_iter):
            self.alpha[t,0] = 0.1
            self.alpha[t,1] = 0.9
        self.alpha = nn.Parameter(self.alpha)

        self.n_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x, get_features=False):
        
        x0 = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        
        hist = [None for _ in range(self.n_history-1)] + [x0]

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

        out = F.adaptive_max_pool2d(hist[-1], (1,1))[:,:,0,0]
        if get_features:
            return out, self.LOutput(out)
        else:
            return self.LOutput(out)

    def save(self, path):
        data = { "input_shape" : self.input_shape,
                 "n_classes" : self.n_classes,
                 "n_filters" : self.n_filters,
                 "n_iter" : self.n_iter,
                 "n_history" : self.n_history,
                 "bias" : self.bias,
                 "conv_mode" : self.conv_mode,
                 "activ" : self.activ,
                 "pool_strategy" : self.pool_strategy,
                 "state_dict" : self.state_dict()}
        torch.save(data, path)


    @staticmethod
    def fromFile(path):
        map_loc = "cpu" if not torch.cuda.is_available() else None
        data = torch.load(path, map_location=map_loc)
        model = ResThriftyNet(data["input_shape"], data["n_classes"], 
                            data["n_filters"], n_iter=data["iter"], 
                            n_history=data["n_history"], pool_strategy=[0], 
                            activ=data["activ"], conv_mode = data["conv_mode"],
                            bias=data["bias"])
        model.pool_strategy = data["pool_strategy"]
        model.load_state_dict(data["state_dict"])
        return model


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
            freq = pool_strategy[0]
            for i in range(self.n_iter):
                if (i%freq == freq-1):
                    self.pool_strategy[i] = True
        else:
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


class ThriftyNet_3State(nn.Module):
    def __init__(self, input_shape, n_classes, n_history, conv_mode="classic", activ="relu", bias=False):
        super(ThriftyNet_3State, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activ = activ
        self.conv_mode = conv_mode
        self.bias = bias
        self.n_history = n_history

        self.n_filters = "32 - 128 - 256"
        self.pool_strategy = [9, 15, 20, 25, 30]

        self.block1 = ThriftyBlock(32, 10, 5, [9], conv_mode=conv_mode, activ=activ, bias=bias)
        self.block2 = ThriftyBlock(128, 12, 5, [6,11], conv_mode=conv_mode, activ=activ, bias=bias)
        self.block3 = ThriftyBlock(256, 12, 5, [6,11], conv_mode=conv_mode, activ=activ, bias=bias)

        self.LOutput = nn.Linear(256, n_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x0 = F.pad(x, (0, 0, 0, 0, 0, 32-3))
        x1 = self.block1(x0)
        x1 = F.pad(x1, (0, 0, 0, 0, 0, 128-32))
        x2 = self.block2(x1)
        x2 = F.pad(x2, (0, 0, 0, 0, 0, 256-128))
        x3 = self.block3(x2)
        if x3.size()[-1]>1:
            out = F.adaptive_max_pool2d(x3, (1,1))[:,:,0,0]
        else:
            out = x3
        return self.LOutput(out)