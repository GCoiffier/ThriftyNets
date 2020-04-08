import torch
import torch.nn as nn
import torch.nn.functional as F

class ReTanh(nn.Module):
    """
    ReTanh activation function
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(ReTanh, self).__init__()
        self.inplace = inplace
        self.tanh = nn.Tanh(inplace = self.inplace)

    def forward(self, x):
        return x * self.tanh(x)

class Swish(nn.Module):
    """
    Swish activation function
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.sigmoid = nn.sigmoid(inplace = self.inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace
        self.relu = nn.ReLU6(inplace = self.inplace)

    def forward(self, x):
        return x * self.relu(x + 3.0) / 6.0

def get_activ(activ):
    if activ=="relu" :
        return nn.ReLU()
    elif activ=="tanh" :
        return nn.Tanh()
    elif activ=="leaky":
        return nn.LeakyReLU(0.1)
    elif activ=="rrelu" :
        return nn.RReLU()
    elif activ == "celu" :
        return nn.CELU()
    elif activ== "tanhshrink":
        return nn.Tanhshrink()
    elif activ == "swish":
        return Swish()
    elif activ== "hswish":
        return HSwish()
    elif activ== "retanh":
        return ReTanh()
    else :
        raise Exception("Activation '{}' is not recognized".format(activ))
    

class MBConv(nn.Module):
    def __init__(self, in_maps, compress):
        super(MBConv, self).__init__()
        self.conv1 = nn.Conv2d(in_maps, compress, kernel_size = 3, padding = 1, bias = False, groups = compress)
        self.conv2 = nn.Conv2d(compress, in_maps, kernel_size = 1, padding = 0, bias = False)

    def forward(self, x):
        return self.conv2(self.conv1(x))