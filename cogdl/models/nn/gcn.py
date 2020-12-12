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
        output = torch.sparse.mm(adj, support)
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


def drop_edge(adj, adj_values, dropedge_rate, train):
    if train:
        import numpy as np
        n_edge = adj.shape[1]
        _remaining_edges = np.arange(n_edge)
        np.random.shuffle(_remaining_edges)
        _remaining_edges = np.sort(_remaining_edges[:int((1 - dropedge_rate) * n_edge)])
        new_adj = adj[:, _remaining_edges]
        new_adj_values = adj_values[_remaining_edges]
    else:
        new_adj = adj
        new_adj_values = adj_values
    return new_adj, new_adj_values


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
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropedge", type=float, default=0.0)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout, args.num_layers, args.dropedge)

    def __init__(self, nfeat, nhid, nclass, dropout, num_layers, dropedge):
        super(TKipfGCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid))
        for _ in range(num_layers - 2):
            self.gcs.append(GraphConvolution(nhid, nhid))
        self.gcs.append(GraphConvolution(nhid, nclass))
        self.dropout = dropout
        self.dropedge = dropedge  # 0 correspond to no dropedge
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):
        device = x.device
        adj_values = torch.ones(adj.shape[1]).to(device)
        original_adj = adj
        original_adj_values = adj_values
        original_x_shape = x.shape[0]
        for idx, gc_layer in enumerate(self.gcs):
            adj, adj_values = drop_edge(original_adj, original_adj_values, self.dropedge, self.training)
            adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, original_x_shape)
            deg = spmm(adj, adj_values, torch.ones(original_x_shape, 1).to(device)).squeeze()
            deg_sqrt = deg.pow(-1 / 2)
            adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
            x = F.dropout(x, self.dropout, training=self.training)
            x = gc_layer(x, adj, adj_values)
            if idx != len(self.gcs) - 1:
                x = F.relu(x)
        return F.log_softmax(x, dim=-1)

        # # TODO: implement drop edge here.
        # adj, adj_values = drop_edge(original_adj, original_adj_values, self.dropedge, self.training)
        # adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        # deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        # deg_sqrt = deg.pow(-1/2)
        # adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
        #
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc1(x, adj, adj_values))
        # # h1 = x
        # x = F.dropout(x, self.dropout, training=self.training)
        # adj, adj_values = drop_edge(original_adj, original_adj_values, self.dropedge, self.training)
        # adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        # deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        # deg_sqrt = deg.pow(-1 / 2)
        # adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
        # x = self.gc2(x, adj, adj_values)
        #
        # # x = F.relu(x)
        # # x = torch.sigmoid(x)
        # # return x
        # # h2 = x
        # return F.log_softmax(x, dim=-1)
    
    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
