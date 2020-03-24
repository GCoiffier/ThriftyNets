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
import os
import sys
from datasets import get_data_loaders
from models import *
import utils


if __name__ == '__main__':

    parser = utils.args()
    parser.add_argument("-teacher", "--teacher", required=True)
    parser.add_argument("-temperature", "--temperature", type=float, default=10.0)
    parser.add_argument("-distil-weight", "--distil-weight", type=float, default=10.0)
    args = parser.parse_args()
    print(args)
    args.no_shuffle = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_loader, test_loader, metadata = get_data_loaders(args)

    teacher_model = ResNet.fromFile(args.teacher)
    teacher_model.eval()

    model = get_model(args, metadata)

    print("N parameters : ", model.n_parameters)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["state_dict"])

    teacher_model = teacher_model.to(device)
    model = model.to(device)

    scheduler = None
    if args.optimizer=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)
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

    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")

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
            real_loss = F.cross_entropy(output, target)
            
            logits_t = F.softmax(teacher_model(data)/args.temperature, dim=1)
            logits_s = F.log_softmax(output/args.temperature, dim=1)
            kd_loss = kd_loss_fn(logits_s, logits_t)
            
            loss = real_loss + args.distil_weight * kd_loss
            loss.backward()
            
            avg_loss += loss.item()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            acc_score = accuracy / ((1+batch_idx) * args.batch_size)
            tqdm.write(prefix+"Epoch {}/{}, Test_loss : {:.3f}, Test_acc : {:.4f}, Train_Loss : {:.3f}, Train_Acc : {:.4f}, LR : {:.1E}".format(
                        epoch, args.epochs, test_loss, test_acc, avg_loss/(1+batch_idx)/args.batch_size, acc_score, lr))

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