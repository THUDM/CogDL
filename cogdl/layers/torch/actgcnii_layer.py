import math

import torch
import torch.nn as nn

from actnn.layers import QLinear

from cogdl.utils import spmm


class ActGCNIILayer(nn.Module):
    def __init__(self, n_channels, alpha=0.1, beta=1, residual=False):
        super(ActGCNIILayer, self).__init__()
        self.n_channels = n_channels
        self.alpha = alpha
        self.beta = beta
        self.residual = residual
        self.linear = QLinear(n_channels, n_channels)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.n_channels)
        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)

    def forward(self, graph, x, init_x):
        """Symmetric normalization"""
        hidden = spmm(graph, x, actnn=True)
        hidden = (1 - self.alpha) * hidden + self.alpha * init_x
        h = self.beta * self.linear(hidden) + (1 - self.beta) * hidden
        if self.residual:
            h = h + x
        return h
