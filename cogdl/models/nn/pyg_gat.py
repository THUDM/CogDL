import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv

from .. import BaseModel, register_model


@register_model("pyg_gat")
class GAT(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--num-heads", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--lr", type=float, default=0.005)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_heads,
            args.dropout,
        )

    def __init__(self, in_feats, hidden_size, out_feats, num_heads, dropout):
        super(GAT, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.conv1 = GATConv(in_feats, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads, out_feats, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
