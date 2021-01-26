import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import SELayer

from .. import BaseModel, register_model
from .gcn import GraphConvolution
from cogdl.utils import add_remaining_self_loops, symmetric_normalization


@register_model("drgcn")
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
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
        )

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout):
        super(DrGCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList([GraphConvolution(shapes[layer], shapes[layer + 1]) for layer in range(num_layers)])
        self.ses = nn.ModuleList(
            [SELayer(shapes[layer], se_channels=int(np.sqrt(shapes[layer]))) for layer in range(num_layers)]
        )

    def forward(self, x, edge_index):
        x = self.ses[0](x)
        edge_index, edge_weight = add_remaining_self_loops(edge_index)
        edge_weight = symmetric_normalization(x.shape[0], edge_index, edge_weight)
        for se, conv in zip(self.ses[1:], self.convs[:-1]):
            x = F.relu(conv(x, edge_index, edge_weight))
            x = se(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
