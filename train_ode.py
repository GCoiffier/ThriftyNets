# This code was forked from : https://github.com/EmilienDupont/augmented-neural-odes

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import *

from tqdm import tqdm, trange
from tqdm._utils import _term_move_up
prefix = _term_move_up() + '\r'

import random
import time
import os
import sys
from thrifty.ode_models import *
from common.datasets import get_data_loaders
from common import utils
from collections import OrderedDict

def get_and_reset_nfes(ode_model):
    """Returns and resets the number of function evaluations for model."""
    iteration_nfes = 0
    for var in vars(ode_model).items():
        if type(var[1])==OrderedDict and var[1]:
            for blockname, block in list(var[1].items()):
                if type(block)==ODEBlock:
                    iteration_nfes += block.odefunc.nfe
                    # Set nfe count to 0 before backward pass, so we can
                    # also measure backwards nfes
                    block.odefunc.nfe = 0
                elif type(block)==ConvODEFunc:  # If we are using ODEBlock
                    iteration_nfes = block.odefunc.nfe
                    block.odefunc.nfe = 0
    return iteration_nfes


if __name__=="__main__":
    parser = utils.args()
    parser.add_argument("-adjoint", "--adjoint", action="store_true")
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

    model = ConvODENet(device, metadata["input_shape"], metadata["n_classes"], args.filters, activ=args.activ, adjoint=args.adjoint).to(device)
    print("N parameters : ", model.n_parameters)
    
    scheduler = None
    if args.optimizer=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)
        # scheduler = StepLR(optimizer, 100, gamma=0.1)
    elif args.optimizer=="adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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
    test_acc = torch.zeros(len(topk))
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        logger.update({"Epoch" :  epoch, "lr" : lr})
        
        epoch_nfes = 0
        epoch_backward_nfes = 0
        loss = 0
        avg_loss = 0
        accuracies = torch.zeros(len(topk))

        ## TRAINING
        optimizer.zero_grad()
        model.train()
        for batch_idx, (x_batch, y_batch) in tqdm(enumerate(train_loader), 
                                              total=len(train_loader),
                                              position=1, 
                                              leave=False, 
                                              ncols=100,
                                              unit="batch"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)

            iteration_nfes = get_and_reset_nfes(model)
            epoch_nfes += iteration_nfes

            loss = F.cross_entropy(y_pred, y_batch)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            accuracies += utils.accuracy(y_pred, y_batch, topk=topk)
            acc_score = accuracies / (1+batch_idx)

            iteration_backward_nfes = get_and_reset_nfes(model)
            epoch_backward_nfes += iteration_backward_nfes

            tqdm_log = prefix+"Epoch {}/{}, LR: {:.1E}, Train_Loss: {:.3f}, Test_loss: {:.3f}, NFE: {}, bkwdNFE : {}, ".format(
                    epoch, args.epochs, lr,  avg_loss/(1+batch_idx), test_loss, iteration_nfes, iteration_backward_nfes)
            for i,k in enumerate(topk):
                tqdm_log += "Train_acc(top{}): {:.3f}, Test_acc(top{}): {:.3f}, ".format(k, acc_score[i], k, test_acc[i])
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

        if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
            name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc(top1)"]))
            torch.save(model.state_dict(), name)

        logger.log()

    torch.save(model.state_dict(), args.name+".model")
