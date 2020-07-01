import torch
import os
import random
import argparse
import numpy as np

try:
    from ptflops import get_model_complexity_info
    def get_info(model, metadata):
        macs, params = get_model_complexity_info(model, metadata["input_shape"], as_strings=True, print_per_layer_stat=False, verbose=False)
        infos = {
            "parameters" : params,
            "flops" : macs,
        }
        if (hasattr(model, "n_filters")):
            infos["filters"] = model.n_filters
        if (hasattr(model, "pool_stategy")):
            infos["pool_strategy"] = model.pool_strategy
        return infos
        
except:
    def get_info(model, metadata):
        infos = {
            "parameters" : sum(p.numel() for p in model.parameters())
        }
        if (hasattr(model, "n_filters")):
            infos["filters"] = model.n_filters
        if (hasattr(model, "pool_stategy")):
            infos["pool_strategy"] =  model.pool_strategy
        return infos

def args():
    parser = argparse.ArgumentParser(description='ThritfyNets')

    # Model settings
    parser.add_argument('-model', "--model", type=str, default="res_thrifty")
    parser.add_argument('-f', "--filters", type=int, default=128, help="Number of filters in the Thrifty Block.")
    parser.add_argument('-T', "--iter", type=int, default=30, help="Depth (#iterations) of the Thrifty Block")
    parser.add_argument('-activ', "--activ", type=str, default="tanh")
    parser.add_argument('-H', "--history", type=int, default=5)
    parser.add_argument('-bias', "--bias", action="store_true")
    parser.add_argument('-pool', "--pool", nargs="+", type=int, default=[7])
    parser.add_argument('-conv-mode', "--conv-mode", type=str, default="mb1")
    parser.add_argument('-out-mode', '--out-mode', type=str, default="pool", choices=["pool", "flatten"])
    parser.add_argument('-n-params', "--n-params", type=int, default=None)

    # Dataset and data augmentation
    parser.add_argument('-dataset', "--dataset", type=str, default=["cifar10"], nargs="+")
    parser.add_argument("-cutout", "--cutout", type=int, default=0)
    parser.add_argument("-auto-augment", "--auto-augment", action="store_true")

    # Optimizer
    parser.add_argument("-opt", "--optimizer", type=str, default="sgd")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov')
    parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float, help="weight decay")
    parser.add_argument('-bs', '--batch-size', type=int, default=100, help="batch size")
    parser.add_argument('-tbs', '--test-batch-size', type=int, default=100)
    parser.add_argument('-epochs', '--epochs', type=int, default=200)

    # Scheduler
    parser.add_argument('--gamma', type=float, default=0.1, help="LR decay factor")
    parser.add_argument('--steps', type=int, nargs="+", default=[50,100,150])

    # Parallelism
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1, help="")
    parser.add_argument("-gpu", "--gpu-devices", type=int, nargs='+', default=[0], help="")

    # Misc
    parser.add_argument('--checkpoint-freq', type=int, default=0, metavar='N',
                        help='we save the network parameters in a file every checkpoint-freq epochs (default: 0 ie no checkpoints)')
    parser.add_argument("-topk", "--topk", type=int, default=None, nargs="+", help="Accuracy to be measured (top 1 acc, top 5 acc, ...)")
    parser.add_argument("--seed", type=int, default=random.randint(0,1000000),
                        help = "random seed to initialize.")
    parser.add_argument("-name", "--name", type=str, default=None,
                        help="name of the experiment in the result log file")
    parser.add_argument("-resume", "--resume", type=str, default=None)

    return parser

def reduceLR(epoch, gamma, steps=None):
    if steps is None:
        return 1.0
    else:
        times = len([x for x in steps if x<=epoch])
        return gamma**times

class Logger:

    def __init__(self, file_path="no_name.log", verbose=False):
        self.path = file_path
        self.verbose = verbose
        self.logged = {}

    def update(self, update_dict):
        self.logged.update(update_dict)

    def __getitem__(self, key):
        return self.logged[key]

    def __setitem__(self, key, item):
        self.logged[key] = item

    def log(self):
        with open(self.path, "a") as f:
            s = ""
            for k in self.logged.keys():
                s += "{} : {}, ".format(k, self.logged[k])
            s+= "\n"
            if self.verbose:
                print(s)
            f.write(s)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return torch.Tensor(res)