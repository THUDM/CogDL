import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

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
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


@register_model("gcn")
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
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout)

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(TKipfGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):
        device = x.device
        adj_values = torch.ones(adj.shape[1]).to(device)
        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        deg_sqrt = deg.pow(-1/2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj, adj_values))
        # h1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, adj_values)

        # x = F.relu(x)
        # x = torch.sigmoid(x)
        # return x
        # h2 = x
        return F.log_softmax(x, dim=-1)
    
    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
