from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from keras_progbar import Progbar

import random
import os
import numpy as np
from datasets import get_data_loaders
from models import get_model
import utils

import matplotlib.pyplot as plt

class MaskRCNN(nn.Module):
    def __init__(self, input_shape, n_classes, n_filters, T=1, pool_freq = 4):
        super(MaskRCNN, self).__init__()
        self.T = T
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.n_classes = n_classes 
        self.pool_freq = pool_freq
        
        self.Lbn = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(T)])
        self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False)
        self.Loutput = nn.Linear(n_filters, n_classes)
    
        self.residual = nn.Parameter(torch.zeros(T*(T+3)//2)+0.1)

        self.masks = nn.Parameter(torch.zeros((T, n_filters)) + 1.)

        self.n_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x, get_features=False):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        X = [x]

        size = x.size()[-1]
        for i in range(self.T):
            # Compute i-th iteration
            x_i = self.Lconv(X[i])
            x_i = F.relu(x_i)
            start = i*(i+3)//2
            residual_i = F.softmax(self.residual[start:start+i+2], dim=0)
            x_i *= residual_i[0]
            for j in range(len(X)):
                x_i += residual_i[j+1] * X[j]
            x_i = self.Lbn[i](x_i)

            # Apply mask
            x_i = F.relu(self.masks[i,:]).view(1,self.n_filters,1,1)*x_i
            X.append(x_i)

            if i%self.pool_freq == self.pool_freq - 1 and size > 1:
                size //= 2
                for k in range(len(X)-1):
                    X[k] = F.avg_pool2d(X[k], 2)
                X[-1] = F.max_pool2d(X[-1], 2)
        
        size = X[-1].size()[-1]
        out = F.adaptive_max_pool2d(X[-1], (1,1))[:,:,0,0]
        if get_features:
            return out, self.Loutput(out)
        else:
            return self.Loutput(out)

    def save(self, path):
        data = { "iter" : self.T,
                 "input_shape" : self.input_shape,
                 "n_filters" : self.n_filters,
                 "n_classes" : self.n_classes,
                 "layer" : self.Lconv,
                 "pool_freq" : self.pool_freq,
                 "mode" : self.mode,
                 "state_dict" : self.state_dict()}
        torch.save(data, path)


def flop_loss(model):
    s = 0
    size = model.input_shape[-1]
    for i in range(1, model.T):
        s += (size*size)*(F.relu(model.masks[i-1,:])*F.relu(model.masks[i, :])).sum()
        if size>1 and i % model.pool_freq==0:
            size//=2
    return s.item()

def sparsity_loss(model):
    s = 0
    for i in range(model.n_filters):
        s += F.l1_loss(F.relu(model.masks[:,i]), torch.zeros_like(model.masks[:,i]))
    return s

if __name__ == '__main__':

    # ------ Initialization ------


    parser = utils.args()
    parser.add_argument("-lmdb-params", "--lmdb-params", type=float, default=1e-2)
    parser.add_argument("-lmdb-sparse", "--lmdb-sparse", type=float, default=1e-2)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader, metadata = get_data_loaders(args)
    model = MaskRCNN((3,32,32), 10, args.size, args.iter, args.downsampling)

    print("N parameters : ", model.n_parameters)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["state_dict"])

    model = model.to(device)

    params = [model.Lconv.weight, model.Loutput.weight, model.residual]
    for x in model.Lbn:
        params.append(x.weight)
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_mask = optim.SGD([model.masks], lr=args.lr, momentum=args.momentum, weight_decay=0)

    scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, verbose=True, min_lr=args.min_lr)
    # scheduler = StepLR(optimizer, 100, gamma=0.1)
    
    try:
        os.mkdir("logs")
    except:
        pass
    logger = utils.Logger("logs/{}.log".format(args.name))
    # ------ End Initialization ------

    for epoch in range(1, args.epochs + 1):
        logger["Epoch"] = epoch
        print("_"*80 + "\nEpoch {}/{}".format(epoch, args.epochs))
        prog = Progbar(len(train_loader.dataset), width=30)
        print("")

        # ----- Train phase -----
        model.train()
        correct = 0
        ce_loss = 0
        fl_loss = 0
        sp_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            optimizer_mask.zero_grad()

            # Get output for accuracy evalutation
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Compute loss
            ce_loss = F.cross_entropy(output, target)
            fl_loss = flop_loss(model)
            sp_loss = sparsity_loss(model)

            train_loss = ce_loss + args.lmdb_params * fl_loss + args.lmbd_sparse * sp_loss
            train_loss.backward()

            optimizer.step()
            optimizer_mask.step()

            prog.add(args.batch_size, 
                    values = [("CE Loss", ce_loss.item()),
                              ("Flop Loss", fl_loss),
                              ("Sparsity Loss", sp_loss.item()),
                              ("Acc", correct / ((1+batch_idx) * args.batch_size))])

        logger.update({
            "train_CE_loss" : ce_loss.item(),
            "train_flop_loss" : fl_loss, 
            "train_sparse_loss" : sp_loss.item(), 
            "train_acc" : correct / len(train_loader.dataset)
        })

        masks = torch.zeros_like(model.masks)
        masks.copy_(model.masks)
        plt.imsave("logs/masks_e{}.png".format(epoch), masks.detach().numpy(), vmin=0.0, vmax=1.1, cmap="hot")

        print("Deactivated filters : ", (model.masks<=0).sum().item(), "/", np.product(model.masks.size()))

        # ----- Test phase -----
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * test_acc)
        )

        logger.update({
            "test_loss" : test_loss,
            "test_acc" : test_acc
        })
        
        scheduler.step(logger["test_loss"])

        if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
            name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc"]))
            print("Saved checkpoint model as " + name)
            model.save(name)

        logger.log()

    # ----- End of training
    with open("results.txt", "a") as f:
        f.write("Name : {}, Size : {:d}, Parameters : {:d}, Iter : {:d}, Pooling freq : {:d}, Epochs : {:d}, Batch size : {:d}, Seed : {:d}, ".format(
            args.name, args.size, model.n_parameters, args.iter, args.downsampling, args.epochs, args.batch_size, args.seed))

        f.write("Train_loss : {:4f}, Train_acc : {:4f}, Test_loss : {:4f}, Test_acc : {:4f}\n".format(
            logger["train_loss"], logger["train_acc"], logger["test_loss"], logger["test_acc"]
        ))

    model.save(args.name+".model")