import torch
import torch.nn as nn
import torch.nn.functional as F
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
        graph.row_norm()
        x = spmm(graph, x)
        return x

    def forward(self, graph, x):
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
        return x

    def forward(self, graph, x):
        x = self.linear(x)
        x = self.aggr(graph, x)
        return x

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class SAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, normalize=False, aggr="mean"):
        super(SAGELayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.normalize = normalize
        if aggr == "mean":
            self.aggr = MeanAggregator(in_feats, out_feats)
        elif aggr == "sum":
            self.aggr = SumAggregator(in_feats, out_feats)
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        out = self.aggr(graph, x)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out
