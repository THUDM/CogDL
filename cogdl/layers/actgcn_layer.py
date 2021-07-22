import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from actnn.layers import QLinear, QReLU

from .qdropout import QDropout
from cogdl.utils import spmm


class ActGCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.0, activation=None, residual=False, norm=None, bias=True):
        super(ActGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = QLinear(in_features, out_features)
        if dropout > 0:
            self.dropout = QDropout(dropout, inplace=True)
        else:
            self.dropout = None
        if residual:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None

        if activation is not None:
            self.act = QReLU()
        else:
            self.act = None

        if norm is not None:
            if norm == "batchnorm":
                self.norm = nn.BatchNorm1d(out_features)
            elif norm == "layernorm":
                self.norm = nn.LayerNorm(out_features)
            else:
                raise NotImplementedError
        else:
            self.norm = None

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)

    def forward(self, graph, x):
        support = self.linear(x)
        out = spmm(graph, support, actnn=True)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        if self.residual is not None:
            out = out + self.residual(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
