import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from cogdl.utils import get_activation
from cogdl.data import Graph


@register_model("mlp")
class MLP(BaseModel):
    r"""Multilayer perception with normalization

    .. math::
        x^{(i+1)} = \sigma(W^{i}x^{(i)})

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    out_feats : int
        Size of each output sample.
    hidden_dim : int
        Size of hidden layer dimension.
    use_bn : bool, optional
        Apply batch normalization if True, default: `True).
    """

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
        self.norm = norm
        self.activation = get_activation(activation)
        self.act_first = act_first
        self.dropout = dropout
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.mlp = nn.ModuleList(
            [nn.Linear(shapes[layer], shapes[layer + 1], bias=bias) for layer in range(num_layers)]
        )
        if norm is not None and num_layers > 1:
            if norm == "layernorm":
                self.norm_list = nn.ModuleList(nn.LayerNorm(x) for x in shapes[1:-1])
            elif norm == "batchnorm":
                self.norm_list = nn.ModuleList(nn.BatchNorm1d(x) for x in shapes[1:-1])
            else:
                raise NotImplementedError(f"{norm} is not implemented in CogDL.")
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            layer.reset_parameters()
        if self.norm:
            for n in self.norm_list:
                n.reset_parameters()

    def forward(self, x, *args, **kwargs):
        if isinstance(x, Graph):
            x = x.x
        for i, fc in enumerate(self.mlp[:-1]):
            x = fc(x)
            if self.act_first:
                x = self.activation(x)
            if self.norm:
                x = self.norm_list[i](x)

            if not self.act_first:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp[-1](x)
        return x

    def predict(self, data):
        return self.forward(data.x)
