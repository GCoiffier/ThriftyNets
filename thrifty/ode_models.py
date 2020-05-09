from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from .modules import *

MAX_NUM_STEPS = 1000

class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.

    Parameters
    ----------
    device : torch.device

    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.

    is_conv : bool
        If True, treats odefunc as a convolutional model.

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, device, odefunc, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)


class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.
    This code was forked from : https://github.com/EmilienDupont/augmented-neural-odes

    Parameters
    ----------
    device : torch.device

    n_filters : int
        Number of convolutional filters.

    n_classes : int
        Final size of output. Should be the number of classes of the dataset

    activ : string
        activation function
    """
    def __init__(self, device, n_filters, activ="relu"):
        super(ConvODEFunc, self).__init__()
        self.device = device
        self.nfe = 0  # Number of function evaluations
        self.n_filters = n_filters
        #self.conv = MBConv(n_filters, n_filters)
        self.conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(n_filters)
        self.activ = get_activ(activ)

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        self.nfe += 1
        out = self.conv(x)
        out = self.activ(out)
        #out = self.bn(out)
        return out


class ConvODENet(nn.Module):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.
    This code was forked from : https://github.com/EmilienDupont/augmented-neural-odes

    Parameters
    ----------
    device : torch.device

    input_shape : tuple of ints
        Tuple of (channels, height, width).

    n_filters : int
        Number of convolutional filters.

    n_classes : int
        Final size of output. Should be the number of classes of the dataset

    activ : string
        activation function

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, device, input_shape, n_classes, n_filters, activ="tanh", tol=1e-3, adjoint=False):
        super(ConvODENet, self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.tol = tol

        odefunc = ConvODEFunc(device, n_filters, activ)
        self.odeblock1 = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
        self.odeblock2 = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
        self.odeblock3 = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)

        self.Loutput = nn.Linear(self.n_filters, self.n_classes)

    def forward(self, x, return_features=False):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        features = self.odeblock1(x)
        features = F.max_pool2d(features,2)
        features = self.odeblock2(x)
        features = F.max_pool2d(features,2)
        features = self.odeblock3(x)
        features = F.adaptive_max_pool2d(features, (1,1))[:,:,0,0]
        pred = self.Loutput(features)
        if return_features:
            return features, pred
        return pred
