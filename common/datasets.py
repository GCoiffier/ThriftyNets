import torch
from torchvision import datasets, transforms
import os
import csv
import numpy as np
from PIL import Image
from .autoaugment import *

# Datasets are downloaded in your home folder. Change DATA_PATH to change download destination
DATA_PATH = "~/.torch_datasets"

def load_mnist(args, **kwargs):
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(DATA_PATH, 'mnist'), train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(DATA_PATH, 'mnist'), train=False, download=True, transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs
    )

    metadata = {
        "input_shape" : (1,28,28),
        "n_classes" : 10
    }

    return train_loader, test_loader, metadata

def load_fashionMnist(args, **kwargs):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(os.path.join(DATA_PATH, 'fashionMnist'), train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(os.path.join(DATA_PATH, 'fashionMnist'), train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    metadata = {
        "input_shape" : (1,28,28),
        "n_classes" : 10
    }

    return train_loader, test_loader, metadata


def load_cifar10(args):

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    transform_list = []
    transform_list.append(transforms.RandomCrop(32, padding=4))
    transform_list.append(transforms.RandomHorizontalFlip())

    if args.auto_augment:
       transform_list.append(CIFAR10Policy())
    if args.cutout>0:
        transform_list.append(Cutout(args.cutout))
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD ))


    transform_train = transforms.Compose(transform_list)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(DATA_PATH, 'cifar10'), train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(DATA_PATH, 'cifar10'), train=False, download=True, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=True)

    metadata = {
        "input_shape" : (3,32,32),
        "n_classes" : 10
    }

    return train_loader, test_loader, metadata


def load_cifar100(args):
    
    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_list = []
    transform_list.append(transforms.RandomCrop(32, padding=4))
    transform_list.append(transforms.RandomHorizontalFlip())

    if args.auto_augment:
       transform_list.append(CIFAR10Policy())
    if args.cutout>0:
        transform_list.append(Cutout(args.cutout))
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD ))

    transform_train = transforms.Compose(transform_list)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(os.path.join(DATA_PATH, 'cifar100'), train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(os.path.join(DATA_PATH, 'cifar100'), train=False, download=True, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=False
    )

    metadata = {
        "input_shape" : (3,32,32),
        "n_classes" : 100
    }
    return train_loader, test_loader, metadata


def load_svhn(args, **kwargs):
    
    SVHN_MEAN = (0.5, 0.5, 0.5)
    SVHN_STD  = (0.5, 0.5, 0.5)

    transform_list = []
    transform_list.append(transforms.RandomCrop(32, padding=4))

    if args.cutout>0:
        transform_list.append(Cutout(args.cutout))
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(SVHN_MEAN, SVHN_STD ))

    transform_train = transforms.Compose(transform_list)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), train=False, download=True, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=False
    )

    metadata = {
        "input_shape" : (3,32,32),
        "n_classes" : 10
    }
    return train_loader, test_loader, metadata


def load_imageNet(imagenet_path, args, **kwargs):
    traindir = os.path.join(imagenet_path, 'train')
    valdir = os.path.join(imagenet_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.auto_augment:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False)

    metadata = {
        "input_shape" : (3,224,224),
        "n_classes" : 1000
    }

    return train_loader, test_loader, metadata

def get_data_loaders(args, **kwargs):

    dataset_name = args.dataset[0].lower()

    if dataset_name == "mnist":
        return load_mnist(args, **kwargs)

    elif dataset_name == "fashionmnist":
        return load_fashionMnist(args, **kwargs)

    elif dataset_name == "cifar10":
        return load_cifar10(args, **kwargs)

    elif dataset_name == "cifar100":
        return load_cifar100(args, **kwargs)
    
    elif dataset_name == "svhn":
        return load_svhn(args, **kwargs)

    elif dataset_name == "imagenet":
        if len(args.dataset)<2:
            raise Exception("Please specify root folder : `-dataset imagenet path/to/imagenet/folder/")
        return load_imageNet(args.dataset[1], args, **kwargs)

    else :
        raise Exception("Dataset '{}' is no recognized dataset. Could not load any data.".format(args.dataset))
