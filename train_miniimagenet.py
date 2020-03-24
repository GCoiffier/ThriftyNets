from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from models import get_model

from keras_progbar import Progbar

import random
import os
from datasets import get_data_loaders
import utils
import numpy as np

"""
Training using a rotation loss, where input images are rotated by a multiple of 90 degrees, and the 
model has to predict the rotation on top of the label

Useful for imagenet and derivated datasets (like miniimagenet)
"""
if __name__ == '__main__':

    args = utils.parse_args()
    args.dataset = "miniimagenet"
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
        
    optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': rot_model.parameters()}],
                lr=args.lr, 
                momentum=args.momentum, 
                weight_decay=args.weight_decay
            )
    
    lossfn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, 100, gamma=0.2)
    
    try:
        os.mkdir("logs")
    except:
        pass
    logger = utils.Logger("logs/{}.log".format(args.name))

    for epoch in range(1, args.epochs + 1):        
        logger["Epoch"] = epoch
        print("_"*80 + "\nEpoch {}/{}".format(epoch, args.epochs))
        prog = Progbar(len(train_loader.dataset), width=30)

        # Train for one epoch
        rot_model.train()
        model.train()

        correct = rcorrect = 0
        avg_loss = 0
        avg_rloss = 0

        for i, (x,y) in enumerate(train_loader):
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
            rotate_scores =  rot_model(final_feat)

            p1 = torch.argmax(scores,1)
            correct += (p1==y_).sum().item()
            p2 = torch.argmax(rotate_scores,1)
            rcorrect += (p2==a_).sum().item()

            optimizer.zero_grad()
            rloss = lossfn(rotate_scores,a_)
            closs = lossfn(scores, y_)
            loss = closs + rloss
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+closs.data.item()
            avg_rloss = avg_rloss+rloss.data.item()

            prog.add(2*bs if args.mix else bs, 
                    values=[("Loss", avg_loss/float(i+1)), ("Acc", correct/(bs*float(i+1))), 
                            ("Rotate loss", avg_rloss/float(i+1)), ("Rotate Acc", rcorrect/(bs*float(i+1)*4))])            

        logger.update({
            "train_loss" : avg_loss/len(train_loader.dataset),
            "train_rotate_loss" : avg_rloss/len(train_loader.dataset),
            "train_acc" : correct/len(train_loader.dataset),
            "train_rotate_acc" : rcorrect/(len(train_loader.dataset)*4)
            })

        if (args.checkpoint_freq>0 and epoch % args.checkpoint_freq==0) :
            outfile = os.path.join("logs", args.name+'_e{:d}.model'.format(epoch))
            model.save(outfile)
        
        # Model evaluation
        model.eval()
        rot_model.eval()

        with torch.no_grad():
            correct = 0
            rcorrect = 0
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
                rotate_scores =  rot_model(final_feat)
                p1 = torch.argmax(scores,1)
                correct += (p1==y_).sum().item()
                p2 = torch.argmax(rotate_scores,1)
                rcorrect += (p2==a_).sum().item()

            logger["test_acc"] = float(correct)/(len(test_loader.dataset))
            logger["test_rotate_acc"] = float(rcorrect)/(len(test_loader.dataset)*4)
            print("Test Accuracy {:.2f}%, Test Rotate Accuracy : {:.2f}%".format(100*logger["test_acc"], 100*logger["test_rotate_acc"]))
        
        torch.cuda.empty_cache()
        scheduler.step()
        logger.log()

    with open("results.txt", "a") as final_feat:
        final_feat.write("Name : {}, Size : {:d}, Parameters : {:d}, Iter : {:d}, Pooling freq : {:d}, Epochs : {:d}, Batch size : {:d}, Seed : {:d}, ".format(
            args.name, args.size, model.n_parameters, args.iter, args.downsampling, args.epochs, args.batch_size, args.seed))

        final_feat.write("Train_loss : {:4f}, Train_acc : {:4f}, Test_loss : {:4f}, Test_acc : {:4f}\n".format(
            logger["train_loss"], logger["train_acc"], logger["test_loss"], logger["test_acc"]
        ))

    model.save(args.name+".model")