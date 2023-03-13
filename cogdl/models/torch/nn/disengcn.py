import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import DisenGCNLayer
from .. import BaseModel


class DisenGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--K", type=int, nargs="+", default=[16, 8])
        parser.add_argument("--iterations", type=int, default=7)
        parser.add_argument("--tau", type=float, default=1)
        parser.add_argument("--activation", type=str, default="leaky_relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feats=args.num_features,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes,
            K=args.K,
            iterations=args.iterations,
            tau=args.tau,
            dropout=args.dropout,
            activation=args.activation,
        )

    def __init__(self, in_feats, hidden_size, num_classes, K, iterations, tau, dropout, activation):
        super(DisenGCN, self).__init__()
        self.K = K
        self.iterations = iterations
        self.dropout = dropout
        self.activation = activation
        self.num_layers = len(K)

        self.weight = nn.Parameter(torch.Tensor(hidden_size, num_classes))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.reset_parameters()

        shapes = [in_feats] + [hidden_size] * self.num_layers
        self.layers = nn.ModuleList(
            DisenGCNLayer(shapes[i], shapes[i + 1], K[i], iterations, tau, activation) for i in range(self.num_layers)
        )

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.zeros_(self.bias.data)

    def forward(self, graph):
        h = graph.x
        graph.remove_self_loops()
        for layer in self.layers:
            h = layer(graph, h)
            # h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = torch.matmul(h, self.weight) + self.bias
        return out

    def predict(self, data):
        return self.forward(data)
