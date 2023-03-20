import torch.nn as nn

from actnn.layers import QLinear, QReLU, QBatchNorm1d, QDropout


class ActMLP(nn.Module):
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
        super(ActMLP, self).__init__()
        self.norm = norm
        if activation == "relu":
            self.activation = QReLU()
        else:
            self.activation = lambda x: x
        self.act_first = act_first
        if dropout > 0.0:
            self.dropout = QDropout(dropout)
        else:
            self.dropout = None
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.mlp = nn.ModuleList([QLinear(shapes[layer], shapes[layer + 1], bias=bias) for layer in range(num_layers)])
        if norm is not None and num_layers > 1:
            if norm == "batchnorm":
                self.norm_list = nn.ModuleList(QBatchNorm1d(x) for x in shapes[1:-1])
            else:
                raise NotImplementedError(f"{norm} is not implemented in CogDL.")
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            layer.reset_parameters()
        if hasattr(self, "norm_list"):
            for n in self.norm_list:
                n.reset_parameters()

    def forward(self, x):
        for i, fc in enumerate(self.mlp[:-1]):
            x = fc(x)
            if self.act_first:
                x = self.activation(x)
            if self.norm:
                x = self.norm_list[i](x)

            if not self.act_first:
                x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        x = self.mlp[-1](x)
        return x
