import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops


@register_model("pyg_unet")
class UNet(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.92)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.num_layers, args.dropout, args.num_nodes)

    def __init__(self, in_feats, hidden_size, out_feats, num_layers, dropout, num_nodes):
        super(UNet, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_nodes = 0

        self.unet = GraphUNet(
            self.in_feats, self.hidden_size, self.out_feats, depth=3, pool_ratios=[2000 / num_nodes, 0.5], act=F.elu
        )

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(
            edge_index, p=0.2, force_undirected=True, num_nodes=x.shape[0], training=self.training
        )
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.unet(x, edge_index)
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
