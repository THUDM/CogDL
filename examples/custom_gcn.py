import torch.nn.functional as F

from cogdl import experiment
from cogdl.models import BaseModel, register_model
from cogdl.models.nn.gcn import GraphConvolution
from cogdl.utils import add_remaining_self_loops, symmetric_normalization


@register_model("mygcn")
class GCN(BaseModel):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout)

    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_feats, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, out_feats)
        self.dropout = dropout

    def forward(self, x, edge_index):
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_attr = symmetric_normalization(x.shape[0], edge_index, edge_attr)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_attr)
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)


if __name__ == "__main__":
    experiment(task="node_classification", dataset="cora", model="mygcn")
