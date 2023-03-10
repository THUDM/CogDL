import numpy as np
import torch.nn.functional as F

from cogdl.layers import SELayer, GATLayer

from .. import BaseModel


class DrGAT(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.num_classes, args.hidden_size, args.nhead, args.dropout,)

    def __init__(self, num_features, num_classes, hidden_size, num_heads, dropout):
        super(DrGAT, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.conv1 = GATLayer(num_features, hidden_size, nhead=num_heads, attn_drop=dropout)
        self.conv2 = GATLayer(hidden_size * num_heads, num_classes, nhead=1, attn_drop=dropout)
        self.se1 = SELayer(num_features, se_channels=int(np.sqrt(num_features)))
        self.se2 = SELayer(hidden_size * num_heads, se_channels=int(np.sqrt(hidden_size * num_heads)))

    def forward(self, graph):
        x = graph.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.se1(x)
        x = F.elu(self.conv1(graph, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.se2(x)
        x = F.elu(self.conv2(graph, x))
        return x
