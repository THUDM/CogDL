import torch
import torch.nn as nn
import torch.nn.functional as F

from actnn.layers import QLinear, QReLU, QBatchNorm1d, QDropout

from cogdl.utils import spmm


class MeanAggregator(object):
    def __call__(self, graph, x):
        graph.row_norm()
        x = spmm(graph, x, actnn=True)
        return x


class SumAggregator(object):
    def __call__(self, graph, x):
        x = spmm(graph, x, actnn=True)
        return x


class ActSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, normalize=False, aggr="mean", dropout=0.0, norm=None, activation=None):
        super(ActSAGELayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = QLinear(2 * in_feats, out_feats)
        self.normalize = normalize
        self.dropout = dropout
        if aggr == "mean":
            self.aggr = MeanAggregator()
        elif aggr == "sum":
            self.aggr = SumAggregator()
        else:
            raise NotImplementedError

        if dropout > 0:
            self.dropout = QDropout(dropout)
        else:
            self.dropout = None

        if activation is not None:
            self.act = QReLU()
        else:
            self.act = None

        if norm is not None:
            if norm == "batchnorm":
                self.norm = QBatchNorm1d(out_feats)
            else:
                raise NotImplementedError
        else:
            self.norm = None

    def forward(self, graph, x):
        out = self.aggr(graph, x)
        out = torch.cat([x, out], dim=-1)
        out = self.fc(out)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        if self.dropout is not None:
            out = self.dropout(out)
        return out
