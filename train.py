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
from common.callback import *
from common import utils
from common.trainer import Trainer

from common.models.densenet import tiny_densenet
from common.models.resnets_cifar import tiny_resnet

from thrifty.models import get_model, get_model_exact_params
from thrifty.sandbox import resnet18


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

def main():
    parser = utils.args()
    args = parser.parse_args()
    print(args)

    os.makedirs("logs", exist_ok=True)
    torch.manual_seed(args.seed)

    if args.distributed:
        gpu_devices = ','.join([str(i) for i in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        n_gpus = len(args.gpu_devices) 
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        if torch.cuda.is_available():
            main_worker(0, 1, args)
        else:
            main_worker("cpu", 1, args)

        
        
def main_worker(device, world_size, args):

    dataset = get_data_loaders(args) # tuple of form (train_loader, test_loader, metadata)
    metadata = dataset[2]

    if args.distributed:
        dist.init_process_group("nccl", rank=device, world_size=world_size, init_method="tcp://127.0.0.1:3456")

    if args.topk is not None:
        topk = tuple(args.topk)
    else:
        if args.dataset=="imagenet":
            topk=(1,5)
        else:
            topk=(1,)

    """
    model = get_model(args, metadata)
    # In case we want an exact number of parameters
    if args.n_params is not None and args.model not in ["block_thrifty", "blockthrifty"]:
        model, args = get_model_exact_params(model, args, metadata)
    """

    model = tiny_resnet()
    #model = tiny_densenet()

    # Log for parameters, filters and pooling strategy
    info_dict = utils.get_info(model, metadata)
    for key,val in info_dict.items():
        print(key, " : ", val)
    print("")

    if args.name is not None:
        with open("logs/{}.log".format(args.name), "a") as f:
            f.write(str(args))
            for key,val in info_dict.items():
                f.write("{} : {}\n".format(key, val))
            f.write("\n*******\n")
        print("-"*80 + "\n")
    
    # Eventually resume training
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["state_dict"])

    if device!="cpu":
        model = model.cuda(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])


    # Init optimizer and scheduler
    scheduler = None
    if args.optimizer=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        schedule_fun = lambda epoch, gamma=args.gamma, steps=args.steps : utils.reduceLR(epoch, gamma, steps)
        scheduler = LambdaLR(optimizer, lr_lambda= schedule_fun)
    elif args.optimizer=="adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Init data augmenter
    dataaugment = None
    if args.mixup:
        dataaugment = "mixup"
    elif args.cutmix:
        dataaugment = "cutmix"

    trainer = Trainer(device, model, dataset, optimizer, CrossEntropy(), args.name, topk, args.checkpoint_freq, dataaugment)

    if scheduler is not None:
        trainer.callbacks.append(SchedulerCB(scheduler))

    trainer.train(args.epochs)
    torch.save(model.state_dict(), args.name+".model")

if __name__ == '__main__':
    main()