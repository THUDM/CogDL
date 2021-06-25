import torch.nn as nn
import torch.nn.functional as F

from cogdl import experiment
from cogdl.layers import GCNLayer
from cogdl.models import BaseModel, register_model


@register_model("mygcn")
class GCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout)

    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(in_feats, hidden_size)
        self.conv2 = GCNLayer(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        h = F.relu(self.conv1(graph, self.dropout(h)))
        h = self.conv2(graph, self.dropout(h))
        return h


if __name__ == "__main__":
    experiment(task="node_classification", dataset="cora", model="mygcn")
