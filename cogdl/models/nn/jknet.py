import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model

import dgl
import dgl.function as fn
import torch

AGGREGATIONS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'max': torch.max,
}

class GraphConvLayer(torch.nn.Module):
    """Graph convolution layer.
    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    def __init__(self, in_features, out_features, aggregation='sum'):
        super(GraphConvLayer, self).__init__()

        if aggregation not in AGGREGATIONS.keys():
            raise ValueError("'aggregation' argument has to be one of "
                             "'sum', 'mean' or 'max'.")
        self.aggregate = lambda nodes: AGGREGATIONS[aggregation](nodes, dim=1)

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

@register_model("jknet")
class JKNet(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--num-features', type=int)
        parser.add_argument('--num-classes', type=int)
        parser.add_argument('--hidden-size', type=int, default=16)
        parser.add_argument('--hidden-layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--aggregation', type=str, default='sum')
        parser.add_argument('--combine', type=str, default='concat')
        # fmt: on
    
    @classmethod
    def build_model_from_args(cls, args):
        return cls(in_features = args.num_features,
                   out_features = args.num_classes,
                   n_layers = args.hidden_layers, 
                   n_units = args.hidden_size,
                   dropout = args.dropout,
                   aggregation = args.aggregation,
                   combine = args.combine,
                  )

    def __init__(self, in_features, out_features, n_layers=2, n_units=16,
                 dropout=0.5, aggregation='sum', combine='concat'):
        super(JKNet, self).__init__()
        self.n_layers = n_layers
        self.graph = None
        self.combine = combine

        self.gconv0 = GraphConvLayer(in_features, n_units, aggregation)
        self.dropout0 = torch.nn.Dropout(dropout)
        for i in range(1, self.n_layers):
            setattr(self, 'gconv{}'.format(i),
                    GraphConvLayer(n_units, n_units, aggregation))
            setattr(self, 'dropout{}'.format(i), torch.nn.Dropout(dropout))
        if self.combine == 'concat':
            self.last_linear = torch.nn.Linear(n_layers * n_units, out_features)
        elif self.combine == 'maxpool':
            self.last_linear = torch.nn.Linear(n_units, out_features)

    def forward(self, x, adj):
        layer_outputs = []
        if self.graph is None:
            self.graph = dgl.DGLGraph().to(x.device)
            self.graph.add_nodes(x.shape[0])
            self.graph.add_edges(adj[0, :], adj[1, :])  

        for i in range(self.n_layers):
            dropout = getattr(self, 'dropout{}'.format(i))
            gconv = getattr(self, 'gconv{}'.format(i))
            x = dropout(F.relu(gconv(self.graph, x)))
            layer_outputs.append(x)

        if self.combine == 'concat':
            h = torch.cat(layer_outputs, dim=1)
        elif self.combine == 'maxpool':
            h = torch.stack(layer_outputs, dim=0)
            h = torch.max(h, dim=0)[0]
        return F.log_softmax(self.last_linear(h), dim=1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
