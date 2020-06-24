import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *

import random
import time
import os
import sys

from common.datasets import get_data_loaders
from common import utils
from common.trainer import Trainer
from common.callback import Callback
from common.losses import *

from thrifty.models import get_model, get_model_exact_params

"""
Callback for exponentially increasing thhe temperature of the alpha loss
"""
class AlphaCallback(Callback):

    def __init__(self, alph):
        self.alph = alph

    def callOnEndForward(self, trainer):
        trainer.temperature *= (1 + self.alph)

class AlphaLoss(LossFun):
    
    name = "AlphaLoss"

    def call(self, output, target, trainer):
        temp = trainer.temperature
        x = trainer.model.Lblock.alpha.data
        loss = x*x*(1-x)*(1-x)
        loss = torch.sum(temp*loss)
        return loss

if __name__ == '__main__':
    os.makedirs("logs", exist_ok=True)

    parser = utils.args()
    parser.add_argument("-alpha", "--alpha", type=float, default = 1.5e-4)
    parser.add_argument("-st", "--starting-temp", type=float, default = 3e-4)
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    dataset = get_data_loaders(args)
    metadata = dataset[2]

    if args.topk is not None:
        topk = tuple(args.topk)
    else:
        if args.dataset=="imagenet":
            topk=(1,5)
        else:
            topk=(1,)

    model = get_model(args, metadata)

    CONV_WEIGHT_BACKUP1 = model.Lblock.Lconv.conv1.weight.data
    CONV_WEIGHT_BACKUP2 = model.Lblock.Lconv.conv2.weight.data

    if args.n_params is not None and args.model not in ["block_thrifty", "blockthrifty"]:
        model, args = get_model_exact_params(model, args, metadata)
       
    # Log for parameters, filters and pooling strategy
    n_parameters = sum(p.numel() for p in model.parameters())
    print("N parameters : ", n_parameters)
    if (hasattr(model, "n_filters")):
        print("N filters : ", model.n_filters)
    if (hasattr(model, "pool_stategy")):
        print("Pool strategy : ", model.pool_strategy)
    
    model = model.to(device)

    if args.resume is None:
        # First phase of training
        # Init optimizer and scheduler
        scheduler = None
        if args.optimizer=="sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            schedule_fun = lambda epoch, gamma=args.gamma, steps=args.steps : utils.reduceLR(epoch, gamma, steps)
            scheduler = LambdaLR(optimizer, lr_lambda= schedule_fun)
        elif args.optimizer=="adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.name is not None:
            with open("logs/{}.log".format(args.name), "a") as f:
                f.write(str(args))
                f.write("\nParameters : {}".format(n_parameters))
                if hasattr(model, "n_filters"):
                    f.write("\nFilters : {}".format(model.n_filters))
                else:
                    f.write("\nFilters : _ ")
                f.write("\n*******\n")
            print("-"*80 + "\n")

        trainer1 = Trainer(device, model, dataset, optimizer, [CrossEntropy(), AlphaLoss()], scheduler, name=args.name, topk=topk, checkpointFreq=args.checkpoint_freq)
        trainer1.temperature = args.starting_temp
        trainer1.callbacks.append(AlphaCallback(args.alpha))
        trainer1.train(args.epochs)
        
        torch.save(model.state_dict(), args.name+".model")

    else: # arg.resume is not None
        model.load_state_dict(torch.load(args.resume))

    print("-"*80)
    print("Binarize and fine tune\n")
    print("")
    FROZEN_ALPHA = (model.Lblock.alpha.data > 0.2).float().to(device)

    with open("logs/{}.log".format(args.name), "a") as f:
        f.write("*******\nFine tuning after binarization\n*******\n")
    model.Lblock.alpha.data = FROZEN_ALPHA
    model.Lblock.alpha.requires_grad = False

    # Beginning of second training phase
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    schedule_fun = lambda epoch, gamma=args.gamma, steps=args.steps : utils.reduceLR(epoch, gamma, steps)
    scheduler = LambdaLR(optimizer, lr_lambda= schedule_fun)

    trainer2 = Trainer(device, model, dataset, optimizer, CrossEntropy(), scheduler, name=args.name, topk=topk, checkpointFreq=args.checkpoint_freq)
    trainer2.train(args.epochs, args.epochs)

    print("\n"+"-"*80)
    print("Train again from scratch with same initialization\n")
    print("")
    with open("logs/{}.log".format(args.name), "a") as f:
        f.write("*******\nTrain from scratch with same init\n*******\n")
    
    # Reinitialize model
    model = get_model(args, metadata).to(device)
    model.Lblock.alpha.data = FROZEN_ALPHA
    model.Lblock.Lconv.conv1.weight.data = CONV_WEIGHT_BACKUP1
    model.Lblock.Lconv.conv2.weight.data = CONV_WEIGHT_BACKUP2
    model.Lblock.alpha.requires_grad = False

    # Beginning of third training phase
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    schedule_fun = lambda epoch, gamma=args.gamma, steps=args.steps : utils.reduceLR(epoch, gamma, steps)
    scheduler = LambdaLR(optimizer, lr_lambda= schedule_fun)

    trainer3 = Trainer(device, model, dataset, optimizer, CrossEntropy(), scheduler, name=args.name, topk=topk, checkpointFreq=args.checkpoint_freq)
    trainer3.train(args.epochs, args.epochs)

    print("\n"+"-"*80)
    print("Train again from scratch, another init\n")
    print("")
    with open("logs/{}.log".format(args.name), "a") as f:
        f.write("*******\nTrain from scratch another init\n*******\n")
    
    # Reinitialize model
    model = get_model(args, metadata).to(device)
    model.Lblock.alpha.data = FROZEN_ALPHA
    model.Lblock.alpha.requires_grad = False

    # Beginning of third training phase
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    schedule_fun = lambda epoch, gamma=args.gamma, steps=args.steps : utils.reduceLR(epoch, gamma, steps)
    scheduler = LambdaLR(optimizer, lr_lambda= schedule_fun)

    trainer4 = Trainer(device, model, dataset, optimizer, CrossEntropy(), scheduler, name=args.name, topk=topk, checkpointFreq=args.checkpoint_freq)
    trainer4.train(args.epochs, args.epochs)
