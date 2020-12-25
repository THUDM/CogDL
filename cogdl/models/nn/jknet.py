import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
from .. import BaseModel, register_model

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
        AGGREGATIONS = {
            'sum': torch.sum,
            'mean': torch.mean,
            'max': torch.max,
        }
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
class JKnet(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num_features", type=int)
        parser.add_argument("--hidden_size", type=int, default=64)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument('--aggregation', type=str, default="sum")
        parser.add_argument('--n_layers', type=int, default=2)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features, 
            args.hidden_size, 
            args.num_classes, 
            args.dropout, 
            args.aggregation,
            args.n_layers
            )

    def __init__(self, num_features, hidden_size, num_classes, dropout, aggregation, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.gconv0 = GraphConvLayer(num_features, hidden_size, aggregation)
        self.dropout0 = torch.nn.Dropout(dropout)
        for i in range(1, self.n_layers):
            setattr(self, 'gconv{}'.format(i),
                    GraphConvLayer(hidden_size, hidden_size, aggregation))
            setattr(self, 'dropout{}'.format(i), torch.nn.Dropout(0.5))

        self.graph = None
        self.last_linear = torch.nn.Linear(hidden_size, num_classes)
        stdv = 1.0 / math.sqrt(self.last_linear.weight.size(1))
        self.last_linear.weight.data.normal_(-stdv, stdv)
        self.last_linear.bias.data.normal_(-stdv, stdv)

    def preprocess(self, x, adj, device):
        graph = dgl.DGLGraph().to(device)
        graph.add_nodes(x.size(0))
        graph.add_edges(adj[0], adj[1])
        self.graph = graph
        return graph

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)

    def forward(self, x, adj):
        layer_outputs = []
        graph = self.preprocess(x, adj, x.device) if not self.graph else self.graph

        for i in range(self.n_layers):
            dropout = getattr(self, 'dropout{}'.format(i))
            gconv = getattr(self, 'gconv{}'.format(i))
            x = dropout(F.relu(gconv(graph, x)))
            layer_outputs.append(x)

        h = torch.stack(layer_outputs, dim=0)
        h = torch.max(h, dim=0)[0]

        return F.log_softmax(self.last_linear(h), dim=-1)


