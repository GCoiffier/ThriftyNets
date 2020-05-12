import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import get_activ

class MBConv(nn.Module):
    def __init__(self, in_maps, compress, bias=False):
        super(MBConv, self).__init__()
        self.conv1 = nn.Conv2d(in_maps, compress, kernel_size = 3, padding = 1, bias = bias, groups = compress)
        self.conv2 = nn.Conv2d(compress, in_maps, kernel_size = 1, padding = 0, bias = bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))

def Conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class ThriftyBlock(nn.Module):
    def __init__(self, n_filters, n_iter, n_history, pool_strategy, conv_mode="classic", activ="relu", bias=False):
        super(ThriftyBlock, self).__init__()
        self.n_filters = n_filters
        self.n_iter = n_iter
        self.n_history = n_history
        self.activ = activ
        self.conv_mode = conv_mode
        self.bias = bias
        
        self.pool_strategy = [False]*self.n_iter
        assert isinstance(pool_strategy, list) or isinstance(pool_strategy, tuple)
        if len(pool_strategy)==1:
            self.n_pool = 0
            freq = pool_strategy[0]
            for i in range(self.n_iter):
                if (i%freq == freq-1):
                    self.pool_strategy[i] = True
                    self.n_pool +=1
        else:
            self.n_pool = len(pool_strategy)
            for x in pool_strategy:
                self.pool_strategy[x] = True

        self.Lactiv = get_activ(activ)
        self.Lnormalization = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])

        if self.conv_mode=="classic":
            self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        elif self.conv_mode=="mb1":
            self.Lconv = MBConv(n_filters, n_filters, bias=self.bias)
        elif self.conv_mode=="mb2":
            self.Lconv = MBConv(n_filters, n_filters//2, bias=self.bias)
        elif self.conv_mode=="mb4":
            self.Lconv = MBConv(n_filters, n_filters//4, bias=self.bias)
        self.activ = get_activ(activ)
        
        self.alpha = torch.zeros((n_iter, n_history+1))
        for t in range(n_iter):
            self.alpha[t,0] = 0.1
            self.alpha[t,1] = 0.9
        self.alpha = nn.Parameter(self.alpha)

        self.n_parameters = sum(p.numel() for p in self.parameters())

    def out_shape(self,input_shape):
        """
        Computes the output tensor's shape, given a specific input shape
        """
        n_filters, x, y = input_shape
        ds = 2**(self.n_pool)
        return (self.n_filters, x // ds, y // ds)


    def forward(self, x):
        hist = [None for _ in range(self.n_history-1)] + [x]

        for t in range(self.n_iter):
            a = self.Lconv(hist[-1])
            a = self.Lactiv(a)
            a = self.alpha[t,0] * a
            for i, x in enumerate(hist):
                if x is not None:
                    a = a + self.alpha[t,i+1] * x

            a = self.Lnormalization[t](a)
            
            for i in range(1, self.n_history-1):
                hist[i] = hist[i+1]
            hist[self.n_history-1] = a

            if self.pool_strategy[t]:
                for i in range(len(hist)):
                    if hist[i] is not None:
                        hist[i] = F.max_pool2d(hist[i], 2)
        return hist[-1]


class ResNetEmbedder(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        """
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.n_parameters = sum(p.numel() for p in self.parameters())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x
        """
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        """