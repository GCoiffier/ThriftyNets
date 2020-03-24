# -*- coding: utf-8 -*-

import argparse
import os
from os.path import join as pjoin
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import pandas as pd
import collections

from models import *
from logger import Logger
from keras_progbar import Progbar

distortions = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
]

CIFAR_PATH = "~/.torch_datasets/cifar10"
DATA_PATH = "../../../CIFAR-10-C"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

BATCH_SIZE = 10

torch.nn.Module.dump_patches = True

parser = argparse.ArgumentParser(description='Evaluates robustness of a model on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-model', type=str, required=True)

args = parser.parse_args()
print(args)

# /////////////// Model  and Data Setup ///////////////
print('Model setup : {}'.format(args.model))

results = dict([(x, -1.0) for x in distortions])
results["clear"] = -1.0
logger = Logger()
logger.update(results)

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

args.prefetch = 4
testset = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=True, transform=transform_test)
clean_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.prefetch,pin_memory=True)

dataframeStarted = None
dataframe = None

model = RCNN.fromFile(args.model)

if torch.cuda.is_available():
    model.cuda()
model = torch.nn.DataParallel(model, device_ids=[0])
cudnn.benchmark = True

model.eval()
cudnn.benchmark = True
print('Model Loaded')

# /////////////// Clean Baseline ////////////////

# correct = 0
# prog = Progbar(10000)
# print("-"*80)
# print("Clean")
# for batch_idx, (data, target) in enumerate(clean_loader):
#     if torch.cuda.is_available():
#         data = data.cuda()
#     output = model(data)
#     pred = output.argmax(dim=1, keepdim=True)
#     if torch.cuda.is_available():
#         target = target.cuda()
#     correct += pred.eq(target.view_as(pred)).sum().item()
#     prog.add(BATCH_SIZE, values=[("acc", float(correct) / (BATCH_SIZE * (batch_idx+1)) )])
# clean_error = 1 - correct/len(clean_loader.dataset)
# print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))

# logger["clear"] = clean_error

# /////////////// Perturbed scores ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def show_performance(model, distortion_name):
    print("-"*80)
    print(distortion_name)
    with torch.no_grad():
        errs = []
        labels = np.load(pjoin(DATA_PATH, "labels.npy"))
        dataset = np.load( pjoin(DATA_PATH, "{}.npy".format(distortion_name)))
        dataset = np.transpose(dataset,[0,3,1,2])

        for severity in range(0, 5):
            print("Severity : ", severity)
            prog = Progbar(10000)
            torch_data = torch.FloatTensor(dataset[10000*severity:10000*(severity+1)])
            for i in range(3):
                torch_data[:,i,:,:] -= CIFAR10_MEAN[i]
                torch_data[:,i,:,:] /= CIFAR10_STD[i]
            torch_labels = torch.LongTensor(labels[10000*severity:10000*(severity+1)])
            test = torch.utils.data.TensorDataset(torch_data, torch_labels)
            distorted_dataset_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

            correct = 0
            for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                data /= 255
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                if torch.cuda.is_available():
                    target = target.cuda()
                correct += pred.eq(target.view_as(pred)).sum()
                prog.add(BATCH_SIZE, values=[("acc", float(correct) / (BATCH_SIZE * (batch_idx+1)) )])
            percentage = float(correct) / 10000
            errs.append(1 - percentage)

        print('\n=Average', tuple(errs))
        return errs

# /////////////// Display Results ///////////////

error_rates = []
result_dict = dict()
for distortion_name in distortions:
    rate = show_performance(model, distortion_name)
    error_rates.append(np.mean(rate))
    print('Distortion: {:15s}  | Error (%): {:.2f}'.format(distortion_name, 100 * np.mean(rate)))
    for a,b in enumerate(rate):
        result_dict["{}_{}".format(distortion_name,a)] = b
    logger.update(result_dict)

logger.log()