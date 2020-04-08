from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *

from tqdm import tqdm, trange
from tqdm._utils import _term_move_up
prefix = _term_move_up() + '\r'

import random
import os
import sys
from datasets import get_data_loaders
from modules import *
import utils

class RoundNoGradient(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x):
		return x.round()

	@staticmethod
	def backward(ctx, g):
		return g


def min_max_quantize(x, bits):
    assert bits >= 1, bits

    if bits==1:
        return torch.sign(x)

    min_val, max_val = x.min(), x.max()

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.cpu().numpy())
        min_val = float(min_val.data.cpu().numpy())

    if max_val-min_val<1e-8:
        return x

    input_rescale = (x - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    # v = torch.floor(input_rescale * n + 0.5) / n
    v = RoundNoGradient.apply(input_rescale * n + 0.5) / n

    v =  v * (max_val - min_val) + min_val
    return v


def maxWeight(weight):
    #liste_max = []
    #index=0
    maxi = 0

    w = weight
    v = w.view(-1)
    maxi = torch.max(torch.abs(v)).cpu().data.numpy()
    n=0
    while(maxi<1):
        maxi=2
        n+=1
    return n-1


def quantifier(weight, n_bit):
    maxi=maxWeight(weight)
    w = weight.clone().cuda() #self.target_modules[index]
    a = w.shape
    v = torch.zeros(a)
    v = v + pow(2, n_bit-1 + maxi)
    v = v.float()
    v = v.cuda()
    w.data.copy_(w.data/v)
    w = w.int()
    w = w.float()
    w.data.copy_(w.data/v)
    return w

## _____________________________________________________________________________________________

class QuantitizedRCNN(nn.Module):

    def __init__(self, input_shape, n_classes, n_filters, n_iter, n_history, pool_strategy, activ="relu", 
                conv_mode="classic", n_bits_weight=8, n_bits_activ=8, bias=False):
        super(QuantitizedRCNN, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_iter = n_iter
        self.n_history = n_history
        self.activ = activ
        self.conv_mode = conv_mode
        self.bias = bias

        self.n_bits_weight = n_bits_weight
        self.n_bits_activ = n_bits_activ

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
        print(self.pool_strategy)

        self.Lactiv = get_activ(activ)
        self.Lnormalization = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])

        if self.conv_mode=="classic":
            self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        elif self.conv_mode=="mb1":
            self.Lconv = MBConv(n_filters, n_filters)
        elif self.conv_mode=="mb2":
            print(n_filters)
            self.Lconv = MBConv(n_filters, n_filters//2)
        elif self.conv_mode=="mb4":
            self.Lconv = MBConv(n_filters, n_filters//4)

        self.LOutput = nn.Linear(n_filters, n_classes)
        self.activ = get_activ(activ)

        self.alpha = torch.zeros((n_iter, n_history+1))
        for t in range(n_iter):
            self.alpha[t,0] = 0.1
            self.alpha[t,1] = 0.9
        self.alpha = nn.Parameter(self.alpha)

        self.n_parameters = sum(p.numel() for p in self.parameters())
        self.quantize()

    def forward(self, x, get_features=False):
        
        x0 = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        x0 = quantifier(x0, self.n_bits_activ)

        hist = [None for _ in range(self.n_history-1)] + [x0]

        for t in range(self.n_iter):
            a = self.Lconv(hist[-1])
            a = self.Lactiv(a)
            a = self.alpha[t,0] * a
            for i, x in enumerate(hist):
                if x is not None:
                    a = a + self.alpha[t,i+1] * x

            a = self.Lnormalization[t](a)
            a = quantifier(a, self.n_bits_activ)

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

    def quantize(self):
        # /!\ Do not quantize batch normalization
        self.QLconv = self.Lconv.weight.data.clone()
        self.QLOutputW = self.LOutput.weight.data.clone()
        self.QLOutputB = self.LOutput.bias.data.clone()
        self.Qalpha = self.alpha.data.clone()
        
        self.Lconv.weight.data.copy_(quantifier(self.QLconv, self.n_bits_weight))
        self.LOutput.weight.data.copy_(quantifier(self.QLOutputW, self.n_bits_weight))
        self.LOutput.bias.data.copy_(quantifier(self.QLOutputB, self.n_bits_weight))
        self.alpha.data.copy_(quantifier(self.Qalpha, self.n_bits_weight))

    def unquantize(self):
        self.alpha.data.copy_(self.Qalpha)
        self.Lconv.weight.data.copy_(self.QLconv)
        self.LOutput.weight.data.copy_(self.QLOutputW)
        self.LOutput.bias.data.copy_(self.QLOutputB)

## _____________________________________________________________________________________________

if __name__ == '__main__':

    parser = utils.args()
    parser.add_argument("-n-bits-weight", "--n-bits-weight", default=6, type=int)
    parser.add_argument("-n-bits-activ", "--n-bits-activ", default=8, type=int)
    args = parser.parse_args()
    args.model = "quantif"
    args.activ = "relu"
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader, metadata = get_data_loaders(args)
    
    model = QuantitizedRCNN(metadata["input_shape"], metadata["n_classes"], args.size, n_iter=args.iter, n_history=args.history, 
            pool_strategy=args.pool, conv_mode=args.conv_mode, n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)

    print("N parameters : ", model.n_parameters)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["state_dict"])

    model = model.to(device)
    scheduler = None
    if args.optimizer=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)
        # scheduler = StepLR(optimizer, 100, gamma=0.1)
    elif args.optimizer=="adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    try:
        os.mkdir("logs")
    except:
        pass
    logger = utils.Logger("logs/{}.log".format(args.name))

    with open("logs/{}.log".format(args.name), "a") as f:
        f.write(str(args))
        f.write("\n*******\n")

    print("-"*80 + "\n")
    test_loss = 0
    test_acc = 0
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    for epoch in range(1, args.epochs + 1):
        logger.update({"Epoch" :  epoch, "lr" : lr})

        ## TRAINING
        model.train()
        accuracy = 0
        acc_score = 0
        loss = 0
        avg_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), 
                                              total=len(train_loader),
                                              position=1, 
                                              leave=False, 
                                              ncols=100,
                                              unit="batch"):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            avg_loss += loss.item()
            loss.backward()
            
            model.unquantize()
            optimizer.step()
            model.quantize()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            acc_score = accuracy / ((1+batch_idx) * args.batch_size)
            tqdm.write(prefix+"Epoch {}/{}, Test_loss : {:.3f}, Test_acc : {:.4f}, Train_Loss : {:.3f}, Train_Acc : {:.4f}, LR : {:.1E}".format(
                        epoch, args.epochs, test_loss, test_acc, avg_loss/(1+batch_idx), acc_score, lr))

        logger.update({"train_loss" : loss.item(), "train_acc" : acc_score})

        ## TESTING
        test_loss = 0
        test_acc = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        logger.update({ "test_loss" : test_loss, "test_acc" : test_acc })
        
        if scheduler is not None:
            scheduler.step(logger["test_loss"])
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        print()

        if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
            name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc"]))
            model.save(name)

        logger.log()

    model.save(args.name+".model")