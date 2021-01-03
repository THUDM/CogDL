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
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


@register_model("jkn")
class JKN(BaseModel):
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
        parser.add_argument("--n-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--link-type", type=str, default='max_pool', choices=['max_pool', 'concat'])
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.n_layers, args.dropout, args.link_type)

    def __init__(self, nfeat, nhid, nclass, n_layers, dropout, link_type):
        super(JKN, self).__init__()
        self.n_layers = n_layers
        self.link_type = link_type

        self.gconv0 = GraphConvolution(nfeat, nhid)
        self.dropout0 = torch.nn.Dropout(dropout)

        for i in range(1, self.n_layers):
            setattr(self, 'gconv{}'.format(i),
                    GraphConvolution(nhid, nhid))
            setattr(self, 'dropout{}'.format(i), torch.nn.Dropout(0.5))

        if self.link_type is 'max_pool':
            self.last_linear = torch.nn.Linear(nhid, nclass)
        else:
            self.last_linear = torch.nn.Linear(nhid * self.n_layers, nclass)

        self.dropout = dropout
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):
        outputs = []

        device = x.device
        adj_values = torch.ones(adj.shape[1]).to(device)
        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        deg_sqrt = deg.pow(-1 / 2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]

        for i in range(self.n_layers):
            dropout = getattr(self, 'dropout{}'.format(i))
            gconv = getattr(self, 'gconv{}'.format(i))
            x = dropout(F.relu(gconv(x, adj, adj_values)))
            outputs.append(x)

        if self.link_type is 'max_pool':
            h = torch.stack(outputs, dim=0)
            h = torch.max(h, dim=0)[0]
        else:
            h = torch.cat(outputs, dim=1)
        h = self.last_linear(h)
        return F.log_softmax(h, dim=-1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
