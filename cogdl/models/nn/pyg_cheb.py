import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import ChebConv

from .. import BaseModel, register_model


@register_model("chebyshev")
class Chebyshev(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--filter-size", type=int, default=5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.filter_size,
        )

    def __init__(self, in_feats, hidden_size, out_feats, num_layers, dropout, filter_size):
        super(Chebyshev, self).__init__()

        self.num_features = in_feats
        self.num_classes = out_feats
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.filter_size = filter_size
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.convs = nn.ModuleList(
            [ChebConv(shapes[layer], shapes[layer + 1], filter_size) for layer in range(num_layers)]
        )

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
