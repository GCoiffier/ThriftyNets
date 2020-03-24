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
from datasets import get_data_loaders
from models import get_model
import utils
from numpy.random import beta


if __name__ == '__main__':

    parser = utils.args()
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader, metadata = get_data_loaders(args)
    model = get_model(args, metadata)
    print("N parameters : ", model.n_parameters)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["state_dict"])

    model = model.to(device)
    rot_model = nn.Sequential(nn.Linear(model.n_filters,4))
    rot_model = rot_model.to(device)

    if args.optimizer=="sgd":
        optimizer = torch.optim.SGD([
                    {'params': model.parameters()},
                    {'params': rot_model.parameters()}],
                    lr=args.lr, 
                    momentum=args.momentum, 
                    weight_decay=args.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, verbose=True, min_lr=args.min_lr)
    elif args.optimizer=="adam":
        optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': rot_model.parameters()}],
                    lr=args.lr, 
                    momentum=args.momentum, 
                    weight_decay=args.weight_decay
        )
        scheduler = None

    lossfn = nn.CrossEntropyLoss()

    try:
        os.mkdir("logs")
    except:
        pass
    logger = utils.Logger("logs/{}.log".format(args.name))

    with open("logs/{}.log".format(args.name), "a") as f:
        f.write(str(args))
        f.write("\n*******\n")

    print("-"*80 + "\n")

    TOTAL_TRAIN = len(train_loader.dataset)
    TOTAL_TEST = len(test_loader.dataset)

    test_loss = 0
    test_rloss = 0
    test_racc = 0
    test_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        logger["Epoch"] = epoch

        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        logger["lr"] = lr

        # TRAINING
        rot_model.train()
        model.train()
        train_acc = 0
        train_racc = 0
        train_loss = 0
        train_rloss = 0
        for batch_idx, (x,y) in tqdm(enumerate(train_loader), 
                                              total=len(train_loader),
                                              position=1, 
                                              leave=False, 
                                              ncols=100,
                                              unit="batch"):

            x, y = x.to(device), y.to(device)
            bs = x.size(0)
            x_ = []
            y_ = []
            a_ = []
            for j in range(bs):
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                x_ += [x[j], x90, x180, x270]
                y_ += [y[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            x_ = torch.autograd.Variable(torch.stack(x_,0)).to(device)
            y_ = torch.autograd.Variable(torch.stack(y_,0)).to(device)
            a_ = torch.autograd.Variable(torch.stack(a_,0)).to(device)

            final_feat,scores = model.forward(x_, get_features=True)
            rotate_scores = rot_model(final_feat)

            p1 = torch.argmax(scores,1)
            train_acc += (p1==y_).sum().item()
            p2 = torch.argmax(rotate_scores,1)
            train_racc += (p2==a_).sum().item()

            optimizer.zero_grad()
            rloss = lossfn(rotate_scores,a_)
            closs = lossfn(scores, y_)
            loss = closs + rloss
            loss.backward()
            optimizer.step()

            train_loss = train_loss+closs.data.item()
            train_rloss = train_rloss+rloss.data.item()

            cur_size = (1+batch_idx)*args.batch_size

            tqdm.write(prefix+"Epoch {}/{}, Test_Loss : {:.3f}, Test_acc : {:.4f}, \
Test_Rloss : {:.3f}, Test_Racc : {:.2f}, \
Train_Loss : {:.3f}, Train_Acc : {:.4f}, \
Train_Rloss : {:.3f}, Train_Racc : {:.2f}, LR : {:.1E}".format(
                        epoch, args.epochs, test_loss, test_acc, 
                        test_rloss, test_racc,
                        train_loss/cur_size, train_acc/cur_size, 
                        train_rloss/cur_size, train_racc/cur_size, lr))

        
        logger.update({"train_loss" : train_loss/TOTAL_TRAIN, 
                       "train_acc" : train_acc/TOTAL_TRAIN,
                       "train_rot_loss" : train_rloss/(4*TOTAL_TRAIN),
                       "train_rot_acc" : train_racc/(4*TOTAL_TRAIN)})

        model.eval()
        rot_model.eval()

        with torch.no_grad():
            test_acc = 0
            test_racc = 0
            for i,(x,y) in enumerate(test_loader):
                bs = x.size(0)
                x_ = []
                y_ = []
                a_ = []
                for j in range(bs):
                    x90 = x[j].transpose(2,1).flip(1)
                    x180 = x90.transpose(2,1).flip(1)
                    x270 =  x180.transpose(2,1).flip(1)
                    x_ += [x[j], x90, x180, x270]
                    y_ += [y[j] for _ in range(4)]
                    a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

                x_ = torch.autograd.Variable(torch.stack(x_,0)).to(device)
                y_ = torch.autograd.Variable(torch.stack(y_,0)).to(device)
                a_ = torch.autograd.Variable(torch.stack(a_,0)).to(device)

                final_feat,scores = model.forward(x_, True)
                rotate_scores = rot_model(final_feat)

                test_rloss += lossfn(rotate_scores,a_)
                test_loss = lossfn(scores, y_)
                p1 = torch.argmax(scores,1)
                test_acc += (p1==y_).sum().item()
                p2 = torch.argmax(rotate_scores,1)
                test_racc += (p2==a_).sum().item()
            test_loss /= TOTAL_TEST
            test_acc /= TOTAL_TEST
            test_rloss /= 4*TOTAL_TEST
            test_racc /= 4*TOTAL_TEST

        logger.update({ "test_loss" : test_loss,
                        "test_acc" : test_acc, 
                        "test_rot_loss" : test_rloss,
                        "test_rot_acc" : test_racc })
        print()

        if scheduler is not None:
            scheduler.step(logger["test_loss"])
        
        if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
            name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc"]))
            model.save(name)

        logger.log()

    model.save(args.name+".model")