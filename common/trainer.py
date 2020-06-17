import torch
import random
import time
import os
import sys

from .utils import accuracy, Logger
from .callback import *

class Trainer:

    """
    Parameters
    ----------

    'device : torch.device
        The device on which the training occurs (cpu or cuda)

    'model' : torch.nn.Module 
        A torch model to be trained

    'dataset' : tuple
        A tuple containing the data loaders for training, testing, as well as a 
        metadata dictionnary for further information.
        Data loaders can be None.

    'optims' : torch.optim.optimizer.Optimizer | iterable of ...
        The optimizer(s) used for training

    'losses' : function | iterable of ...
        The loss function(s) to be optimized

    'scheduler' : torch.optim.lr_scheduler
        The leraning rate scheduler to eventually use during training. None by default

    'name' : string
        The name of the model and the .log file to write logs at each epoch
    
    'topk' : tuple
        Indicator for which metrics to measure during testing. Default is (1,), that is measuring top1 accuracy.
    """

    def __init__(self, device, model, dataset, optims, losses, scheduler=None, name=None, topk=(1,), checkpointFreq=0):
        self.device = device

        self.model = model
        self.train_data, self.test_data, self.dataset_meta = dataset

        if type(optims) in [list, tuple]:
            self.optims = [x for x in optims]
        else:
            self.optims = [optims]
            

        if type(losses) in [list, tuple]:
            self.losses = [x for x in losses]
        else :
            self.losses = [losses]

        self.scheduler = scheduler
        
        self.name = "unnamed"
        self.logFile = "unnamed.log"
        self.logger = None
        if name is not None:
            assert isinstance(name, str)
            self.name = name
            try:
                os.mkdir("logs")
            except:
                pass
            self.logFile = "logs/{}.log".format(name)
            self.logger = Logger(self.logFile)

        self.checkpointFreq = checkpointFreq
        self.topk = topk

        self.metrics = dict()
        self.metrics["test_loss"] = 0
        self.metrics["train_loss"] = 0
        self.metrics["train_acc"] = torch.zeros(len(self.topk))
        self.metrics["test_acc"] = torch.zeros(len(self.topk))

        self.callbacks = [TqdmCB()]
        if self.name != "unnamed":
            self.callbacks.append(CheckpointCB())
            self.callbacks.append(LoggerCB())
    

    def train(self, Nepochs, epoch_start=1):
        for epoch in range(epoch_start, epoch_start+Nepochs):
            self.metrics["epoch"] = epoch
            self._train_for_one_epoch(epoch)
            self._call_end_train_CB()
            if self.test_data is not None:
                self._test_on_dataset(epoch)
                self._call_end_test_CB()

    def _train_for_one_epoch(self, epoch):
        t0 = time.time()
        self.metrics["lr"] = self.optims[0].state_dict()["param_groups"][0]["lr"]
        self.metrics["train_loss"] = 0
        self.metrics["train_acc"] = torch.zeros(len(self.topk))
        self.model.train()
        for batch_idx, (data, target) in tqdm(enumerate(self.train_data), 
                                            total=len(self.train_data),
                                            position=1, 
                                            leave=False, 
                                            ncols=100,
                                            unit="batch"):

            self.metrics["batch_idx"] = batch_idx
            data, target = data.to(self.device), target.to(self.device)
            for optim in self.optims :
                optim.zero_grad()
            output = self.model(data)
            
            loss = 0
            for lossFun in self.losses:
                loss += lossFun.call(output, target, self)
            loss.backward() 
            
            for optim in self.optims:           
                optim.step()

            self.metrics["train_acc"] += accuracy(output, target, topk=self.topk)
            self.metrics["train_loss"] += loss

            self._call_end_forward_CB()
        self.metrics["epoch_time"] = (time.time() - t0)/60


    def _test_on_dataset(self, epoch):
        self.metrics["test_loss"] = 0
        self.metrics["test_acc"] = torch.zeros(len(self.topk))
        self.model.eval()
        with torch.no_grad():
            for data, target in tqdm(self.test_data,
                                     total=len(self.test_data),
                                     position=1, 
                                     leave=False, 
                                     ncols=100,
                                     unit="batch"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                for lossFun in self.losses:
                    self.metrics["test_loss"] += lossFun.call(output, target, self).sum().item()
                self.metrics["test_acc"] += accuracy(output, target, topk=self.topk)

        self.metrics["test_loss"] /= len(self.test_data.dataset)
        self.metrics["test_acc"] /= len(self.test_data)

    def _call_end_forward_CB(self):
        for cb in self.callbacks:
            cb.callOnEndForward(self)

    def _call_end_train_CB(self):
        for cb in self.callbacks:
            cb.callOnEndTrain(self)

        if self.scheduler is not None:
            self.scheduler.step(self.metrics["test_loss"])
        self.metrics["lr"] = self.optims[0].state_dict()["param_groups"][0]["lr"]

    def _call_end_test_CB(self):
        for cb in self.callbacks:
            cb.callOnEndTest(self)


        
        
        
        