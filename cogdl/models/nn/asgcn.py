import math
import random
import collections
import numpy as np
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model

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

    def forward(self, input, adj):
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

@register_model("asgcn")
class ASGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[64,64,32])
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.sample_size,
        )

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout, sample_size):
        super(ASGCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sample_size = sample_size

        self.w_s0 = Parameter(torch.FloatTensor(num_features))
        self.w_s1 = Parameter(torch.FloatTensor(num_features))

        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList(
            [
                GraphConvolution(shapes[layer], shapes[layer + 1])
                for layer in range(num_layers)
            ]
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.w_s0.data.normal_(-stdv, stdv)
        self.w_s1.data.normal_(-stdv, stdv)

    def set_adj(self, edge_index, num_nodes):
        self.sparse_adj = sparse.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        self.num_nodes = num_nodes
        self.adj = self.compute_adjlist(self.sparse_adj)
        self.adj = torch.tensor(self.adj)

    def compute_adjlist(self, sp_adj, max_degree=32):
        """Transfer sparse adjacent matrix to adj-list format"""
        num_data = sp_adj.shape[0]
        adj = num_data + np.zeros((num_data+1, max_degree), dtype=np.int32)

        for v in range(num_data):
            neighbors = np.nonzero(sp_adj[v, :])[1]
            len_neighbors = len(neighbors)
            if len_neighbors > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
                adj[v] = neighbors
            else:
                adj[v, :len_neighbors] = neighbors

        return adj

    def from_adjlist(self, adj):
        """Transfer adj-list format to sparsetensor"""
        u_sampled, index = torch.unique(torch.flatten(adj), return_inverse=True)

        row = (torch.range(0, index.shape[0]-1) / adj.shape[1]).long().to(adj.device)
        col = index
        values = torch.ones(index.shape[0]).float().to(adj.device)
        indices = torch.cat([row.unsqueeze(1), col.unsqueeze(1)], axis=1).t()
        dense_shape = (adj.shape[0], u_sampled.shape[0])

        support = torch.sparse_coo_tensor(indices, values, dense_shape)

        return support, u_sampled.long()

    def _sample_one_layer(self, x, adj, v, sample_size):
        support, u = self.from_adjlist(adj)


        h_v = torch.sum(torch.matmul(x[v], self.w_s1))
        h_u = torch.matmul(x[u], self.w_s0)
        attention = (F.relu(h_v + h_u) + 1) * (1.0 / sample_size)
        g_u = F.relu(h_u) + 1

        p1 = attention * g_u
        p1 = p1.cpu()

        if self.num_nodes in u:
            p1[u == self.num_nodes] = 0
        p1 = p1 / torch.sum(p1)

        samples = torch.multinomial(p1, sample_size, False)
        u_sampled = u[samples]
            
        support_sampled = torch.index_select(support, 1, samples)

        return u_sampled, support_sampled

    def sampling(self, x, v):
        all_support = [[] for _ in range(self.num_layers)]
        sampled = v
        x = torch.cat((x, torch.zeros(1, x.shape[1]).to(x.device)), dim=0)
        for i in range(self.num_layers - 1, -1, -1):
            cur_sampled, cur_support = self._sample_one_layer(x, self.adj[sampled], sampled, self.sample_size[i])
            all_support[i] = cur_support.to(x.device)
            sampled = cur_sampled

        return x[sampled.to(x.device)], all_support, 0

    def forward(self, x, adj):
        for index, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, adj[index]))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj[-1])
        return F.log_softmax(x, dim=1)
