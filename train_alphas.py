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

from thrifty.models import get_model

"""
Callback for exponentially increasing thhe temperature of the alpha loss
"""
class AlphaCallback(Callback):

    def __init__(self, alph):
        self.alph = alph

    def callOnEndForward(self, trainer):
        trainer.temperature *= (1 + self.alph)


if __name__ == '__main__':

    try:
        os.mkdir("logs")
    except:
        pass

    parser = utils.args()
    parser.add_argument("-alpha", "--alpha", type=float, default = 1e-5)
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
        scheduler = None
        if args.optimizer=="sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)
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
        trainer1.train(3*args.epochs//4)

    else: # arg.resume is not None
        model.load_state_dict(torch.load(args.resume))

    print("-"*80)
    print("BINARIZATION\n")
    with open("logs/{}.log".format(args.name), "a") as f:
        f.write("*******\nShortcut Binarization\n*******\n")
    model.Lblock.alpha.data = (model.Lblock.alpha.data > 0.2).float().to(device)
    print(model.Lblock.alpha)
    model.Lblock.alpha.requires_grad = False

    # Beginning of second training phase
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)

    trainer2 = Trainer(device, model, dataset, optimizer, CrossEntropy(), scheduler, name=args.name, topk=topk, checkpointFreq=args.checkpoint_freq)
    trainer2.train(args.epochs//4, 3*args.epochs//4)
    torch.save(model.state_dict(), args.name+".model")
