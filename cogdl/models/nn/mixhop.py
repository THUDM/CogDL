from cogdl.utils import add_remaining_self_loops
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from cogdl.layers import MixHopLayer

from .. import BaseModel, register_model


@register_model("mixhop")
class MixHop(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--dropout", type=float, default=0.7)
        parser.add_argument("--layer1-pows", type=int, nargs="+", default=[200, 200, 200])
        parser.add_argument("--layer2-pows", type=int, nargs="+", default=[20, 20, 20])
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.dropout,
            args.layer1_pows,
            args.layer2_pows,
        )

    def __init__(self, num_features, num_classes, dropout, layer1_pows, layer2_pows):
        super(MixHop, self).__init__()

        self.dropout = dropout

        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        layer_pows = [layer1_pows, layer2_pows]

        shapes = [num_features] + [sum(layer1_pows), sum(layer2_pows)]

        self.mixhops = nn.ModuleList(
            [
                MixHopLayer(shapes[layer], [0, 1, 2], layer_pows[layer])
                for layer in range(len(layer_pows))
            ]
        )
        self.fc = nn.Linear(shapes[-1], num_classes)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        for mixhop in self.mixhops:
            x = F.relu(mixhop(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
