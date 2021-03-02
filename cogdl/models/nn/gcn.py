import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, symmetric_normalization, spmm


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr=None):
        support = torch.mm(x, self.weight)
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1]).to(x.device)
        out = spmm(edge_index, edge_attr, support)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


@register_model("gcn")
class TKipfGCN(BaseModel):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
        in_features (int) : Number of input features.
        out_features (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.num_layers, args.dropout)

    def __init__(self, in_feats, hidden_size, out_feats, num_layers, dropout):
        super(TKipfGCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList([GraphConvolution(shapes[i], shapes[i + 1]) for i in range(num_layers)])
        self.num_layers = num_layers
        self.dropout = dropout

    def get_embeddings(self, x, edge_index):
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_attr = symmetric_normalization(x.shape[0], edge_index, edge_attr)

        h = x
        for i in range(self.num_layers - 1):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.layers[i](h, edge_index, edge_attr)
        return h

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_attr = symmetric_normalization(x.shape[0], edge_index, edge_weight)
        h = x
        for i in range(self.num_layers):
            h = self.layers[i](h, edge_index, edge_attr)
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
        return h

    def predict(self, data):
        if hasattr(data, "norm_aggr"):
            return self.forward(data.x, data.edge_index, data.norm_aggr)
        else:
            return self.forward(data.x, data.edge_index)
