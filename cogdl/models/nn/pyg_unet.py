import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj

from .. import BaseModel, register_model


@register_model("unet")
class UNet(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
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
        super(UNet, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.unet = GraphUNet(num_features, hidden_size, num_classes, depth=3, pool_ratios=[0.5, 0.5])

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=x.shape[0],
                                    training=self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
