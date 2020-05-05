import torch
import torch.nn as nn


class MeanAggregator(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, improved=False, cached=False, bias=True
    ):
        super(MeanAggregator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.linear = nn.Linear(in_channels, out_channels, bias)

    @staticmethod
    def norm(x, edge_index):
        # here edge_index is already a sparse tensor
        deg = torch.sparse.sum(edge_index, 1)
        deg_inv = deg.pow(-1).to_dense()

        x = torch.matmul(edge_index, x)
        #  print(x,deg_inv)
        x = x.t() * deg_inv
        #  x：512*dim, edge_weight：256*512

        return x.t()

    def forward(self, x, edge_index, edge_weight=None, bias=True):
        """"""
        x = self.linear(x)
        x = self.norm(x, edge_index)
        return x

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
