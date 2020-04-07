import torch
import os
import random
import argparse

def args():
    parser = argparse.ArgumentParser(description='ArchiSearch Guillaume IMT Atlantique')

    # Model settings
    parser.add_argument('-model', "--model", type=str, default="res_thrifty")
    parser.add_argument('-size', "--size", type=int, default=64)
    parser.add_argument('-iter', "--iter", type=int, default=25)
    parser.add_argument("-activ", "--activ", type=str, default="tanh")
    parser.add_argument("-history", "--history", type=int, default=5)
    parser.add_argument("-bias", "--bias", action="store_true")
    parser.add_argument("-pool", "--pool", nargs="+", type=int, default=[5])
    parser.add_argument("-conv-mode", "--conv-mode", type=str, default="classic")

    # Dataset and data augmentation
    parser.add_argument('-dataset', "--dataset", type=str, default="cifar10")
    parser.add_argument("-cutout", "--cutout", type=int, default=0)
    parser.add_argument("-auto-augment", "--auto-augment", action="store_true")

    # Optimizer
    parser.add_argument("-opt", "--optimizer", type=str, default="sgd")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument("-epochs", "--epochs", type=int, default=200)

    # Scheduler
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5, help="LR decay factor")

    # Misc
    parser.add_argument('--checkpoint-freq', type=int, default=0, metavar='N',
                        help='we save the network parameters in a file every checkpoint-freq epochs (default: 0 ie no checkpoints)')
    parser.add_argument("--seed", type=int, default=random.randint(0,1000000),
                        help = "random seed to initialize.")
    parser.add_argument("-name", "--name", type=str, default="unnamed",
                        help="name of the experiment in the result txt file")
    parser.add_argument("-resume", "--resume", type=str, default=None)

    return parser

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
