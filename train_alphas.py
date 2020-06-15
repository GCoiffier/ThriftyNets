from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *

from tqdm import tqdm, trange
from tqdm._utils import _term_move_up
prefix = _term_move_up() + '\r'

import random
import time
import os
import sys

from common.datasets import get_data_loaders
from common import utils

from thrifty.models import get_model
from thrifty.modules import MBConv

"""
Training procedure using two optimizers :
- one for the weights (Conv, BatchNorm and FC layers of a Thrifty)
- one for the alphas (ie, architecture/shortcuts of the Thrifty)

Alphas are penalized with a loss that changes over time in order to force them to {0; 1} at the end
"""
def alpha_loss(x, temp=1.0):
    loss = x*x*(1-x)*(1-x)
    loss = torch.sum(temp*loss)
    return loss

if __name__ == '__main__':

    parser = utils.args()
    parser.add_argument("-alpha", "--alpha", type=float, default = 1e-5)
    parser.add_argument("-st", "--starting-temp", type=float, default = 3e-4)
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader, metadata = get_data_loaders(args)

    if args.topk is not None:
        topk = tuple(args.topk)
    else:
        if args.dataset=="imagenet":
            topk=(1,5)
        else:
            topk=(1,)

    model = get_model(args, metadata)
    if args.n_params is not None and args.model not in ["block_thrifty", "blockthrifty"]:
        n = model.n_parameters
        if n<args.n_params:
            while n<args.n_params:
                args.filters += 1
                model = get_model(args, metadata)
                n = model.n_parameters
        if n>args.n_params:
            while n>args.n_params:
                args.filters -= 1 
                model = get_model(args,metadata)
                n = model.n_parameters

    print("N parameters : ", model.n_parameters)
    print("N filters : ", model.n_filters)
    print("Pool strategy : ", model.pool_strategy)

    model = model.to(device)
    
    if args.resume is None:
        # First phase of training
        scheduler = None
        if args.optimizer=="sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)
        elif args.optimizer=="adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        try:
            os.mkdir("logs")
        except:
            pass
        logger = utils.Logger("logs/{}.log".format(args.name))

        with open("logs/{}.log".format(args.name), "a") as f:
            f.write(str(args))
            f.write("\nParameters : " + str(model.n_parameters))
            f.write("\nFilters : " + str(model.n_filters))
            f.write("\n*******\n")

        print("-"*80 + "\n")
        test_loss = 0
        temperature = args.starting_temp
        test_acc = torch.zeros(len(topk))

        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        for epoch in range(1, 3*(args.epochs + 1)//4+1):

            t0 = time.time()
            logger.update({"Epoch" :  epoch, "lr" : lr})

            ## TRAINING
            model.train()
            accuracies = torch.zeros(len(topk))
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

                alLoss = alpha_loss(model.Lblock.alpha, temperature)
                temperature *= (1 + args.alpha)
                alLoss.backward()
                
                optimizer.step()

                accuracies += utils.accuracy(output, target, topk=topk)
                acc_score = accuracies / (1+batch_idx)

                tqdm_log = prefix+"Epoch {}/{}, LR: {:.1E}, Train_Loss: {:.3f}, Test_loss: {:.3f}, ".format(epoch, args.epochs, lr, avg_loss/(1+batch_idx), test_loss)
                for i,k in enumerate(topk):
                    tqdm_log += "Train_acc(top{}): {:.3f}, Test_acc(top{}): {:.3f}".format(k, acc_score[i], k, test_acc[i])
                tqdm.write(tqdm_log)

            logger.update({"epoch_time" : (time.time() - t0)/60 })
            logger.update({"train_loss" : loss.item()})
            for i,k in enumerate(topk):
                logger.update({"train_acc(top{})".format(k) : acc_score[i]})

            ## TESTING
            test_loss = 0
            test_acc = torch.zeros(len(topk))
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    test_acc += utils.accuracy(output, target, topk=topk)

            test_loss /= len(test_loader.dataset)
            test_acc /= len(test_loader)

            logger.update({"test_loss" : test_loss})
            for i,k in enumerate(topk):
                logger.update({"test_acc(top{})".format(k) : test_acc[i]})
            
            if scheduler is not None:
                scheduler.step(logger["test_loss"])
            
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            print()

            logger.update({"params" : model.n_parameters})
            model = model.to(device)
            optim.param_groups = model.parameters()

            if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
                name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc(top1)"]))
                torch.save(model.state_dict(), name)

            logger.log()
    # End of first training phase

    else: # arg.resume is not None
        model.load_state_dict(torch.load(args.resume)["state_dict"])

    print("-"*80 + "\n")
    print("\nBINARIZATION\n\n")
    with open("logs/{}.log".format(args.name), "a") as f:
        f.write("\n*******\n")
        f.write("\nShortcut Binarization\n")
        f.write("\n*******\n")
    model.Lblock.alpha.data = (model.Lblock.alpha.data > 1e-4).float().to(device)
    model.Lblock.alpha.requires_grad = False

    # Beginning of second training phase
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)

    test_loss = 0
    test_acc = torch.zeros(len(topk))
    lr = optimizer.state_dict()["param_groups"][-1]["lr"]
    for epoch in range(3*(args.epochs + 1)//4 + 1,  args.epochs + 1):

        t0 = time.time()
        logger.update({"Epoch" :  epoch, "lr" : lr})

        ## TRAINING
        model.train()
        accuracies = torch.zeros(len(topk))
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
            optimizer.step()

            accuracies += utils.accuracy(output, target, topk=topk)
            acc_score = accuracies / (1+batch_idx)

            tqdm_log = prefix+"Epoch {}/{}, LR: {:.1E}, Train_Loss: {:.3f}, Test_loss: {:.3f}, ".format(epoch, args.epochs, lr, avg_loss/(1+batch_idx), test_loss)
            for i,k in enumerate(topk):
                tqdm_log += "Train_acc(top{}): {:.3f}, Test_acc(top{}): {:.3f}".format(k, acc_score[i], k, test_acc[i])
            tqdm.write(tqdm_log)

        logger.update({"epoch_time" : (time.time() - t0)/60 })
        logger.update({"train_loss" : loss.item()})
        for i,k in enumerate(topk):
            logger.update({"train_acc(top{})".format(k) : acc_score[i]})

        ## TESTING
        test_loss = 0
        test_acc = torch.zeros(len(topk))
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                test_acc += utils.accuracy(output, target, topk=topk)

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader)

        logger.update({"test_loss" : test_loss})
        for i,k in enumerate(topk):
            logger.update({"test_acc(top{})".format(k) : test_acc[i]})
        
        if scheduler is not None:
            scheduler.step(logger["test_loss"])
        
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        print()

        logger.update({"params" : model.n_parameters})
        model = model.to(device)
        optim.param_groups = model.parameters()

        if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
            name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc(top1)"]))
            torch.save(model.state_dict(), name)

        logger.log()



    torch.save(model.state_dict(), args.name+".model")