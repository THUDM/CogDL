import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.modules.conv import GCNConv

from . import register_model, BaseModel


@register_model("us_gcn")
class UnsupervisedGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--hidden-size", type=int, default=16)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_layers,
            args.dropout,
        )

    def __init__(self, num_features, hidden_size, num_layers, dropout):
        super(UnsupervisedGCN, self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        shapes = [num_features] + [hidden_size] * (num_layers - 1)
        self.convs = nn.ModuleList(
            [
                GCNConv(shapes[layer], shapes[layer + 1], cached=True)
                for layer in range(num_layers - 1)
            ]
        )

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x 
