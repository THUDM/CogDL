from .. import BaseModel
from cogdl.layers import MLP as MLPLayer
from cogdl.data import Graph


class MLP(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=16)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.activation,
            args.norm,
            args.act_first if hasattr(args, "act_first") else False,
        )

    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        dropout=0.0,
        activation="relu",
        norm=None,
        act_first=False,
        bias=True,
    ):
        super(MLP, self).__init__()
        self.nn = MLPLayer(in_feats, out_feats, hidden_size, num_layers, dropout, activation, norm, act_first, bias)

    def forward(self, x):
        if isinstance(x, Graph):
            x = x.x
        return self.nn(x)

    def predict(self, data):
        return self.forward(data.x)
