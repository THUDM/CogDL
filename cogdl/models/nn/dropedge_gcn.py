import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, spmm, spmm_adj, add_self_loops


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, withloop=False, withbn=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)
        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
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
        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)
        if self.bias is not None:
            output = output + self.bias
        if self.bn is not None:
            output = self.bn(output)
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


def bingge_norm_adj(adj, adj_values, num_nodes):
    adj, adj_values = add_self_loops(adj, adj_values, 1, num_nodes)
    deg = spmm(adj, adj_values, torch.ones(num_nodes, 1).to(adj.device)).squeeze()
    deg_sqrt = deg.pow(-1 / 2)
    adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
    row, col = adj[0], adj[1]
    mask = row != col
    adj_values[row[mask]] += 1
    return adj, adj_values


def aug_norm_adj(adj, adj_values, num_nodes):
    adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, num_nodes)
    deg = spmm(adj, adj_values, torch.ones(num_nodes, 1).to(adj.device)).squeeze()
    deg_sqrt = deg.pow(-1 / 2)
    adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
    return adj, adj_values


def get_normalizer(normalization):
    normalizer_dict = dict(AugNorm=aug_norm_adj,
                           BinggeNorm=bingge_norm_adj)
    if not normalization in normalizer_dict:
        raise NotImplementedError
    return normalizer_dict[normalization]


@register_model("dropedge_gcn")
class DropEdgeGCN(BaseModel):
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
        parser.add_argument("--num-layers", type=int, default=0)
        parser.add_argument("--withloop", action="store_true", default=False)
        parser.add_argument("--withbn", action="store_true", default=False)
        parser.add_argument("--dropedge", type=float, default=0.0)
        parser.add_argument("--normalization", type=str, default="AugNorm")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout, args.num_layers, args.withloop,
                   args.withbn, args.dropedge, args.normalization)

    def __init__(self, nfeat, nhid, nclass, dropout, num_layers, withloop, withbn, dropedge, normalization):
        super(DropEdgeGCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.input_gc = GraphConvolution(nfeat, nhid, withloop=withloop, withbn=withbn)
        self.gcs = nn.ModuleList()
        for _ in range(num_layers):
            self.gcs.append(GraphConvolution(nhid, nhid, withloop=withloop, withbn=withbn))
        self.output_gc = GraphConvolution(nhid, nclass, withloop=withloop, withbn=withbn)
        self.dropout = dropout
        self.dropedge = dropedge  # 0 correspond to no dropedge
        self.normalization = normalization
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):
        device = x.device
        adj_values = torch.ones(adj.shape[1]).to(device)
        original_adj = adj  # (2, 9104)
        original_adj_values = adj_values  # (9104)
        original_x_shape = x.shape[0]

        adj, adj_values = drop_edge(original_adj, original_adj_values, self.dropedge, self.training)
        # Add support to different normalizers
        adj, adj_values = get_normalizer(self.normalization)(adj, adj_values, original_x_shape)
        '''
        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, original_x_shape)
        # print(adj.shape, adj_values.shape)  # dropedge=0., (2, 12431), (12431)
        deg = spmm(adj, adj_values, torch.ones(original_x_shape, 1).to(device)).squeeze()
        # print(max(adj[0]), max(adj[1]))  # 3326, 3326
        # print(deg.shape)  # Tensor, (3327)
        # print(max(deg), min(deg))  # 100, 1
        # print(adj_values.shape, max(adj_values), min(adj_values))  # (12431), every element is 1
        deg_sqrt = deg.pow(-1 / 2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
        # print(adj_values.shape, max(adj_values), min(adj_values))  # (12431), max: 1, min: 0.01
        '''

        # Input layer
        x = self.input_gc(x, adj, adj_values)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # Mid layers
        for idx, gc_layer in enumerate(self.gcs):
            x = gc_layer(x, adj, adj_values)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        # Output layer
        x = self.output_gc(x, adj, adj_values)
        return F.log_softmax(x, dim=-1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
