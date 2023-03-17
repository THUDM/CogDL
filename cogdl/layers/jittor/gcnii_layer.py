import math
import jittor
from jittor import nn, Module

from cogdl.utils import spmm


class GCNIILayer(Module):
    def __init__(self, n_channels, alpha=0.1, beta=1, residual=False):
        super(GCNIILayer, self).__init__()
        self.n_channels = n_channels
        self.alpha = alpha
        self.beta = beta
        self.residual = residual
        self.linear = nn.Linear(n_channels, n_channels)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.n_channels)
        jittor.init.uniform_(self.linear.weight, -stdv, stdv)

    def execute(self, graph, x, init_x):
        """Symmetric normalization"""
        hidden = spmm(graph, x)
        hidden = (1 - self.alpha) * hidden + self.alpha * init_x
        h = self.beta * self.linear(hidden) + (1 - self.beta) * hidden
        if self.residual:
            h = h + x
        return h
