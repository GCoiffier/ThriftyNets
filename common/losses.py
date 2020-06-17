import torch
import torch.nn.functional as F

class LossFun:

    name = "NULL"

    def __init__(self):
        pass

    def call(self, output, target, trainer):
        raise NotImplementedError


class CrossEntropy(LossFun):
    
    name = "CrossEntropy"

    def call(self, output, target, trainer):
        return F.cross_entropy(output, target)


class AlphaLoss(LossFun):
    
    name = "AlphaLoss"

    def call(self, output, target, trainer):
        temp = trainer.temperature
        x = trainer.model.Lblock.alpha.data
        loss = x*x*(1-x)*(1-x)
        loss = torch.sum(temp*loss)
        return loss