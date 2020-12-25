import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import dgl
import dgl.function as fn

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
        self.self_loop_w = torch.nn.Linear(in_features, out_features)
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
            return output + self.self_loop_w(input) + self.bias
        else:
            return output + self.self_loop_w(input)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GraphConvLayer(torch.nn.Module):
    """Graph convolution layer.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    AGGREGATIONS = {
        'sum': torch.sum,
        'mean': torch.mean,
        'max': torch.max,
    }
    def __init__(self, in_features, out_features, aggregation='sum'):
        super(GraphConvLayer, self).__init__()

        if aggregation not in self.AGGREGATIONS.keys():
            raise ValueError("'aggregation' argument has to be one of "
                             "'sum', 'mean' or 'max'.")
        self.aggregate = lambda nodes: self.AGGREGATIONS[aggregation](nodes, dim=1)

        self.linear = torch.nn.Linear(in_features, out_features)
        self.self_loop_w = torch.nn.Linear(in_features, out_features)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, graph, x):
        graph.ndata['h'] = x
        graph.update_all(
            fn.copy_src(src='h', out='msg'),
            lambda nodes: {'h': self.aggregate(nodes.mailbox['msg'])})
        h = graph.ndata.pop('h')
        h = self.linear(h)
        return h + self.self_loop_w(x) + self.bias

@register_model("xss-jknet")
class JKnet(BaseModel):
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
        parser.add_argument('--n-layers', type=int, default=2)
        parser.add_argument('--aggregation', type=str, default="sum")
        parser.add_argument('--sparse', type=eval, default=True)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout, args.aggregation, args.n_layers, args.sparse)

    def __init__(self, nfeat, nhid, nclass, dropout, aggregation, n_layers, sparse=True):
        super().__init__()
        self.sparse = sparse
        self.n_layers = n_layers
        
        if self.sparse:
            self.gc0 = GraphConvolution(nfeat, nhid)
            for i in range(1, self.n_layers):
                setattr(self, 'gc{}'.format(i), GraphConvolution(nhid, nhid))
            self.dropout = dropout
        else:

            self.gconv0 = GraphConvLayer(nfeat, nhid, aggregation)
            self.dropout0 = torch.nn.Dropout(dropout)
            for i in range(1, self.n_layers):
                setattr(self, 'gconv{}'.format(i),
                        GraphConvLayer(nhid, nhid, aggregation))
                setattr(self, 'dropout{}'.format(i), torch.nn.Dropout(0.5))

            self.graph = None
        self.last_linear = torch.nn.Linear(nhid, nclass)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.last_linear.weight.size(1))
        self.last_linear.weight.data.normal_(-stdv, stdv)
        self.last_linear.bias.data.normal_(-stdv, stdv)

    def preprocessing(self, x, adj, device):
        graph = dgl.DGLGraph().to(device)
        graph.add_nodes(x.size(0))
        graph.add_edges(adj[0], adj[1])
        self.graph = graph
        return graph

    def forward(self, x, adj):
        layer_outputs = []
        if self.sparse:
            device = x.device
            adj_values = torch.ones(adj.shape[1]).to(device)
            adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
            deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
            deg_sqrt = deg.pow(-1/2)
            adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]

            # deg_ = 1 / deg
            # adj_values = adj_values * deg_[adj[1]]

            for i in range(self.n_layers):

                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(getattr(self, 'gc{}'.format(i))(x, adj, adj_values))
                x = F.dropout(x, self.dropout, training=self.training)
                layer_outputs.append(x)

            h = torch.stack(layer_outputs, dim=0)
            h = torch.max(h, dim=0)[0]

        # x = F.relu(x)
        # x = torch.sigmoid(x)
        # return x
        # h2 = x

        else:
            graph = self.preprocessing(x, adj, x.device) if not self.graph else self.graph

            for i in range(self.n_layers):
                dropout = getattr(self, 'dropout{}'.format(i))
                gconv = getattr(self, 'gconv{}'.format(i))
                x = dropout(F.relu(gconv(graph, x)))
                layer_outputs.append(x)

            h = torch.stack(layer_outputs, dim=0)
            h = torch.max(h, dim=0)[0]

        return F.log_softmax(self.last_linear(h), dim=-1)
    
    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
