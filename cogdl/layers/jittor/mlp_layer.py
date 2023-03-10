from cogdl.backend import BACKEND
from jittor import nn, Module
from cogdl.utils import get_activation

class MLP(Module):
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
        Apply batch normalization if True, default: `True`.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        dropout=0.0,
        activation="sigmoid",
        norm=None,
        act_first=False,
        bias=True,
    ):
        super(MLP, self).__init__()
        self.norm = norm
        self.activation = get_activation(activation)
        self.act_first = act_first
        self.dropout = dropout
        self.output_dim = out_feats
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.mlp = nn.ModuleList(
            [nn.Linear(shapes[layer], shapes[layer + 1], bias=bias) for layer in range(num_layers)]
        )
        if norm is not None and num_layers > 1:
            if norm == "layernorm":
                self.norm_list = nn.ModuleList(nn.LayerNorm(x) for x in shapes[1:-1])
            elif norm == "batchnorm":
                self.norm_list = nn.ModuleList([nn.BatchNorm1d(x) for x in shapes[1:-1]])
            else:
                raise NotImplementedError(f"{norm} is not implemented in CogDL.")
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            nn.init.xavier_uniform_(layer.weight)

    def execute(self, x):
        for k, layer in enumerate(self.mlp):
            x = layer(x)
            if k < len(self.mlp)-1:
                if self.act_first:
                    x = self.activation(x)
                if self.norm:
                    x = self.norm_list[k](x)
                if not self.act_first:
                    x = self.activation(x)
                x = nn.dropout(x, p=self.dropout, is_train=self.is_train)
        return x
