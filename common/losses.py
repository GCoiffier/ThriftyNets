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