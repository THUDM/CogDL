import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, spmm, spmm_adj

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

    def forward(self, input, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1]).float().to(input.device)
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_attr,
            (input.shape[0], input.shape[0]),
        ).to(input.device)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


@register_model("mygcn")
class TKipfGCN(BaseModel):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size1", type=int, default=64)
        parser.add_argument("--hidden-size2", type=int, default=32)
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size1, args.hidden_size2,  args.num_classes, args.dropout)

    def __init__(self, in_feats, hidden_size1, hidden_size2, out_feats, dropout):
        super(TKipfGCN, self).__init__()

        self.gc1 = GraphConvolution(in_feats, hidden_size1)
        self.gc2 = GraphConvolution(hidden_size1, hidden_size2)
        self.gc3 = GraphConvolution(hidden_size2, out_feats)
        self.dropout = dropout
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):
        device = x.device
        adj_values = torch.ones(adj.shape[1]).to(device)
        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        deg_sqrt = deg.pow(-1 / 2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj, adj_values))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj, adj_values))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj, adj_values)
        return x
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)