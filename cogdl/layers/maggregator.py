import torch
import torch.nn as nn
from cogdl.utils import spmm


class MeanAggregator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(MeanAggregator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached_result = None

        self.linear = nn.Linear(in_channels, out_channels, bias)

    @staticmethod
    def norm(graph, x):
        # here edge_index is already a sparse tensor

        graph.row_norm()
        x = spmm(graph, x)
        #  x：512*dim, edge_weight：256*512

        return x

    def forward(self, graph, x):
        """"""
        x = self.linear(x)
        x = self.norm(graph, x)
        return x

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class SumAggregator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SumAggregator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached_result = None

        self.linear = nn.Linear(in_channels, out_channels, bias)

    @staticmethod
    def aggr(graph, x):
        x = spmm(graph, x)
        #  x：512*dim, edge_weight：256*512
        return x

    def forward(self, graph, x):
        """"""
        x = self.linear(x)
        x = self.aggr(graph, x)
        return x

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)
