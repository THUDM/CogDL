import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import spmm, get_activation, get_norm_layer


class MeanAggregator(object):
    def __call__(self, graph, x):
        graph.row_norm()
        x = spmm(graph, x)
        return x


class SumAggregator(object):
    def __call__(self, graph, x):
        x = spmm(graph, x)
        return x


class MaxAggregator(object):
    def __init__(self):
        from cogdl.operators.scatter_max import scatter_max

        self.scatter_max = scatter_max

    def __call__(self, graph, x):
        x = self.scatter_max(graph.row_indptr.int(), graph.col_indices.int(), x)
        return x


class SAGELayer(nn.Module):
    def __init__(
        self, in_feats, out_feats, normalize=False, aggr="mean", dropout=0.0, norm=None, activation=None, residual=False
    ):
        super(SAGELayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(2 * in_feats, out_feats)
        self.normalize = normalize
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if aggr == "mean":
            self.aggr = MeanAggregator()
        elif aggr == "sum":
            self.aggr = SumAggregator()
        elif aggr == "max":
            self.aggr = MaxAggregator()
        else:
            raise NotImplementedError

        if activation is not None:
            self.act = get_activation(activation, inplace=True)
        else:
            self.act = None

        if norm is not None:
            self.norm = get_norm_layer(norm, out_feats)
        else:
            self.norm = None

        if residual:
            self.residual = nn.Linear(in_features=in_feats, out_features=out_feats)
        else:
            self.residual = None

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

        if self.residual:
            out = out + self.residual(x)

        if self.dropout is not None:
            out = self.dropout(out)

        return out
