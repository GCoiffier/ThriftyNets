from __future__ import print_function
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
from common.losses import *
from common import utils
from common.trainer import Trainer

from thrifty.models import get_model, get_model_exact_params

if __name__ == '__main__':
    os.makedirs("logs", exist_ok=True)
    
    parser = utils.args()
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    dataset = get_data_loaders(args) # tuple of form (train_loader, test_loader, metadata)
    metadata = dataset[2]

    if args.topk is not None:
        topk = tuple(args.topk)
    else:
        if args.dataset=="imagenet":
            topk=(1,5)
        else:
            topk=(1,)

    model = get_model(args, metadata)

    # In case we want an exact number of parameters
    if args.n_params is not None and args.model not in ["block_thrifty", "blockthrifty"]:
        model, args = get_model_exact_params(model, args, metadata)

    # Log for parameters, filters and pooling strategy
    n_parameters = sum(p.numel() for p in model.parameters())
    print("N parameters : ", n_parameters)
    if (hasattr(model, "n_filters")):
        print("N filters : ", model.n_filters)
    if (hasattr(model, "pool_stategy")):
        print("Pool strategy : ", model.pool_strategy)
    
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

    # Eventually resume training
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["state_dict"])
    model = model.to(device)

    # Init optimizer and scheduler
    scheduler = None
    if args.optimizer=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        schedule_fun = lambda epoch, gamma=args.gamma, steps=args.steps : utils.reduceLR(epoch, gamma, steps)
        scheduler = LambdaLR(optimizer, lr_lambda= schedule_fun)
    elif args.optimizer=="adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    trainer = Trainer(device, model, dataset, optimizer, CrossEntropy(), scheduler, args.name, topk, args.checkpoint_freq)

    trainer.train(args.epochs)
    torch.save(model.state_dict(), args.name+".model")
