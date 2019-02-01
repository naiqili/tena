import torch
import torch.nn.functional as F
from torch.nn.modules import Module

__all__ = ['BasicReLU', 'TenaReLU', 'TenaSigmoid']

class BasicReLU(Module):
    def __init__(self, negative_slope=0, inplace=False):
        super(BasicReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)
        
    def update(self, epoch):
        print('BasicReLU updated: %d (no effect)' % epoch)
        pass

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + str(self.negative_slope) \
            + inplace_str + ')'

class TenaReLU(Module):
    def __init__(self, negative_slope=1e-2, inplace=False):
        super(TenaReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)
        
    def update(self, epoch):
        #print('TenaReLU updated: %d' % epoch)
        self.negative_slope = 1.0 / (1 + epoch)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + str(self.negative_slope) \
            + inplace_str + ')'

class TenaSigmoid(Module):
    def __init__(self, theta = 0, rate = 0.01):
        super(TenaSigmoid, self).__init__()
        self.theta = theta
        self.rate = rate
        self.sig = torch.nn.Sigmoid()
        
    def forward(self, x):
        return self.theta * self.sig(x) + (1 - self.theta) * (x + 0.5)
        
    def update(self, epoch):
        #print('TenaSigmoid updated: %d' % epoch)
        self.theta = min(1, epoch * self.rate)