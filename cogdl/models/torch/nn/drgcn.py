import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import SELayer, GCNLayer

from .. import BaseModel


class DrGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=16)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.norm,
            args.activation,
        )

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout, norm=None, activation="relu"):
        super(DrGCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList(
            [
                GCNLayer(shapes[layer], shapes[layer + 1], activation=activation, norm=norm)
                for layer in range(num_layers - 1)
            ]
        )
        self.convs.append(GCNLayer(shapes[-2], shapes[-1]))
        self.ses = nn.ModuleList(
            [SELayer(shapes[layer], se_channels=int(np.sqrt(shapes[layer]))) for layer in range(num_layers)]
        )

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        x = self.ses[0](x)
        for se, conv in zip(self.ses[1:], self.convs[:-1]):
            x = conv(graph, x)
            x = se(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](graph, x)
        return x

    def predict(self, graph):
        return self.forward(graph)
