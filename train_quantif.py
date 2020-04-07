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
    #max = bin_list(weight)
    #j=0
    #for index in range(self.num_of_params):
    w = weight.clone() #self.target_modules[index]
    a = w.shape
    v = torch.zeros(a)
    v = v + pow(2, n_bit-1 + maxi)
    v = v.float()
    v = v.cuda()
    w.data.copy( w.datav)
    w = w.int()
    w = w.float()
    w.data.copy_(w.data/v)
    return w

## _____________________________________________________________________________________________

class QuantitizedRCNN(nn.Module):

    def __init__(self, input_shape, n_classes, n_filters, n_iter=20, pool_freq = 4, mode="3", n_bits_weight=6, n_bits_activ=8):
        super(QuantitizedRCNN, self).__init__()
        self.n_iter = n_iter
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.n_classes = n_classes 
        self.pool_freq = pool_freq
        self.mode = mode
        self.n_hist = int(mode)
        self.n_bits_weight = n_bits_weight
        self.n_bits_activ = n_bits_activ

        self.activ = nn.ReLU()

        self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.LOutput = nn.Linear(n_filters, n_classes)
        self.residual = nn.Parameter(torch.FloatTensor(n_iter, self.n_hist+1).uniform_(0.9,1.1) )
        self.Lbn = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])

        self.n_parameters = sum(p.numel() for p in self.parameters())
        self.quantize()

    def forward(self, x, get_features=False):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        x = quantifier(x, self.n_bits_activ)

        hist = [None for _ in range(self.n_hist-1)] + [x]
        size = x.size()[-1]

        for n in range(self.n_iter):
            cur = self.Lconv(hist[-1])
            cur = self.activ(cur)
            cur = self.residual[n,0] * cur
            for i, x in enumerate(hist):
                if x is not None:
                    cur = cur + self.residual[n,i+1] * x
            cur = self.Lbn[n](cur)
            cur = uantifier(cur, self.n_bits_activ)

            if n%self.pool_freq == self.pool_freq - 1 and size > 1:
                size //= 2
                cur = F.max_pool2d(cur, 2)
                for i in range(len(hist)):
                    if hist[i] is not None:
                        hist[i] = F.max_pool2d(hist[i], 2)
            
            for i in range(1, self.n_hist-1):
                hist[i] = hist[i+1]
            hist[self.n_hist-1] = cur
          
        size = hist[-1].size()[-1]
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
        self.Qresidual = self.residual.data.clone()
        
        self.Lconv.weight.data.copy_(quantifier(self.QLconv, self.n_bits_weight))
        self.LOutput.weight.data.copy_(quantifier(self.QLOutputW, self.n_bits_weight))
        self.LOutput.bias.data.copy_(quantifier(self.QLOutputB, self.n_bits_weight))
        self.residual.data.copy_(quantifier(self.Qresidual, self.n_bits_weight))

    def unquantize(self):
        self.residual.data.copy_(self.Qresidual)
        self.Lconv.weight.data.copy_(self.QLconv)
        self.LOutput.weight.data.copy_(self.QLOutputW)
        self.LOutput.bias.data.copy_(self.QLOutputB)

    def save(self, path):
        data = { "iter" : self.n_iter,
                 "input_shape" : self.input_shape,
                 "n_filters" : self.n_filters,
                 "n_classes" : self.n_classes,
                 "mode" : self.mode,
                 "n_bits_activ" : self.n_bits_activ,
                 "n_bits_weight" : self.n_bits_weight,
                 "pool_freq" : self.pool_freq,
                 "state_dict" : self.state_dict()}
        torch.save(data, path)


    @staticmethod
    def fromFile(path):
        map_loc = "cpu" if not torch.cuda.is_available() else None
        data = torch.load(path, map_location=map_loc)
        model = QuantitizedRCNN(data["input_shape"], data["n_classes"], data["n_filters"], n_iter=data["iter"], 
                                pool_freq=data["pool_freq"], mode=data["mode"],
                                n_bits_activ=data["n_bits_activ"], n_bits_weight=data["n_bits_weight"])
        model.load_state_dict(data["state_dict"])
        return model

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
    
    model = QuantitizedRCNN(metadata["input_shape"], metadata["n_classes"], args.size, n_iter=args.iter, pool_freq=args.pool[0], mode=args.mode,
            n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)

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