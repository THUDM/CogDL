
import jittor.nn as nn
from cogdl.layers import GATLayer
# from .. import BaseModel
from cogdl.models import BaseModel
from cogdl.datasets.planetoid_data import CoraDataset
from cogdl.datasets.customized_data import NodeDataset, generate_random_graph

class GAT(BaseModel):
    r"""The GAT model from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
        alpha (float) : Coefficient of leaky_relu.
        nheads (int) : Number of attention heads.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--attn-drop", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--last-nhead", type=int, default=2)
        parser.add_argument("--norm", type=str, default=None)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.attn_drop,
            args.alpha,
            args.nhead,
            args.residual,
            args.last_nhead,
            args.norm,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_features,
        num_layers,
        dropout,
        attn_drop,
        alpha,
        nhead,
        residual,
        last_nhead,
        norm=None,
    ):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.attentions.append(
            GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm)
        )
        for i in range(num_layers - 2):
            self.attentions.append(
                GATLayer(
                    hidden_size * nhead,
                    hidden_size,
                    nhead=nhead,
                    attn_drop=attn_drop,
                    alpha=alpha,
                    residual=residual,
                    norm=norm,
                )
            )
        self.attentions.append(
            GATLayer(
                hidden_size * nhead, out_features, attn_drop=attn_drop, alpha=alpha, nhead=last_nhead, residual=False,
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def execute(self, graph):
        x = graph.x
        for i, layer in enumerate(self.attentions):
            x = nn.dropout(x, p=self.dropout, is_train=self.is_train)
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = nn.elu(x)
        return x

    def predict(self, graph):
        return self.execute(graph)