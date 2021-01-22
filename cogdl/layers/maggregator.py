import torch
import torch.nn as nn


class MeanAggregator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(MeanAggregator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached_result = None

        self.linear = nn.Linear(in_channels, out_channels, bias)

    @staticmethod
    def norm(x, adj_sp):
        # here edge_index is already a sparse tensor
        deg = torch.sparse.sum(adj_sp, 1)
        deg_inv = deg.pow(-1).to_dense()

        x = torch.spmm(adj_sp, x)
        #  print(x,deg_inv)
        x = x.t() * deg_inv
        #  x：512*dim, edge_weight：256*512

        return x.t()

    def forward(self, x, adj_sp):
        """"""
        x = self.linear(x)
        x = self.norm(x, adj_sp)
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
    def aggr(x, adj):
        x = torch.spmm(adj, x)
        #  x：512*dim, edge_weight：256*512
        return x

    def forward(self, x, adj):
        """"""
        x = self.linear(x)
        x = self.aggr(x, adj)
        return x

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)
