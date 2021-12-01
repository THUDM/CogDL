import torch.nn as nn

try:
    from cogdl.layers.actgcn_layer import ActGCNLayer
except Exception:
    print("Please install the actnn library first.")
    exit(1)

from .. import BaseModel


class ActGCN(BaseModel):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
        in_features (int) : Number of input features.
        out_features (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.activation,
            args.residual,
            args.norm,
            args.rp_ratio,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout,
        activation="relu",
        residual=False,
        norm=None,
        rp_ratio=1,
    ):
        super(ActGCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList(
            [
                ActGCNLayer(
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                    residual=residual if i != num_layers - 1 else None,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                    rp_ratio=rp_ratio,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def embed(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers - 1):
            h = self.layers[i](graph, h)
        return h

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        return h
