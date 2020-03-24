import torch
from torchvision import datasets, transforms
import os
import csv
import numpy as np
from PIL import Image
from autoaugment import *

DATA_PATH = "~/.torch_datasets"
MINI_IMAGENET_PATH = os.path.join(DATA_PATH, "miniImagenet")

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


class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root_path, train=True, transform=None):
        self.train = train

        unique_labels = os.listdir(root_path)
        self.labels = {}
        for i,l in enumerate(unique_labels):
            self.labels[l] = i
        
        self.all = []
        for l in unique_labels:
            p = os.path.join(MINI_IMAGENET_PATH, l)
            self.all += [[l,x] for x in os.listdir(p)]

        self.split = int(0.9*len(self.all))
        self.train_ex = self.all[:self.split]
        self.test_ex = self.all[self.split:]
        np.random.shuffle(self.train_ex)
        self.transform = transform

    def __getitem__(self, i):
        if self.train :
            label, img_name = self.train_ex[i]
        else:
            label, img_name = self.test_ex[i]
        path = os.path.join(MINI_IMAGENET_PATH, label)
        path = os.path.join(path, img_name)
        img = Image.open(path).convert("RGB")
        return (self.transform(img), self.labels[label])

    def __len__(self):
        return len(self.train_ex) if self.train else len(self.test_ex)


def load_miniImageNet(args, **kwargs):
    
    if args.auto_augment:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ImageNetPolicy(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    train_loader = torch.utils.data.DataLoader(
        MiniImageNet(MINI_IMAGENET_PATH, train=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        MiniImageNet(MINI_IMAGENET_PATH, train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    metadata = {
        "input_shape" : (3,84,84),
        "n_classes" : 64
    }

    return train_loader, test_loader, metadata


def get_data_loaders(args, **kwargs):

    dataset_name = args.dataset.lower()

    if dataset_name == "mnist":
        return load_mnist(args, **kwargs)

    elif dataset_name == "fashionmnist":
        return load_fashionMnist(args, **kwargs)

    elif dataset_name == "cifar10":
        return load_cifar10(args, **kwargs)

    elif dataset_name == "cifar100":
        return load_cifar100(args, **kwargs)
       
    elif dataset_name == "miniimagenet":
        return load_miniImageNet(args, **kwargs)

    else :
        raise Exception("Dataset '{}' is no recognized dataset. Could not load any data.".format(args.dataset))
