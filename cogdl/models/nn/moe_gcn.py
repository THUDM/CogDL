import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model
from cogdl.utils import spmm, get_activation


from fmoe import FMoETransformerMLP


class CustomizedMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(self, d_model, d_inner, dropout, moe_num_expert=64, moe_top_k=2):
        activation = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, top_k=moe_top_k, activation=activation
        )

        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(d_model)

    def forward(self, inp):
        ##### positionwise feed-forward
        core_out = super().forward(inp)
        core_out = self.dropout(core_out)

        ##### residual connection + batch normalization
        output = self.bn_layer(inp + core_out)

        return output


class GraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.0, residual=False, activation=None, norm=None, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.act = get_activation(activation)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if residual:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None

        if norm is not None:
            if norm == "batchnorm":
                self.norm = nn.BatchNorm1d(out_features)
            elif norm == "layernorm":
                self.norm = nn.LayerNorm(out_features)
            else:
                raise NotImplementedError
        else:
            self.norm = None

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, graph, x):
        support = torch.mm(x, self.weight)
        out = spmm(graph, support)
        if self.bias is not None:
            out = out + self.bias

        if self.residual is not None:
            res = self.residual(x)
            if self.act is not None:
                res = self.act(res)
            out = out + res

        if self.dropout is not None:
            out = self.dropout(out)

        if self.norm is not None:
            out = self.norm(out)

        return out


class GraphConvBlock(nn.Module):
    def __init__(self, conv_func, conv_params, out_feats, dropout=0.0, in_feats=None, residual=False):
        super(GraphConvBlock, self).__init__()

        self.graph_conv = conv_func(**conv_params)
        self.pos_ff = CustomizedMoEPositionwiseFF(out_feats, out_feats * 2, dropout, moe_num_expert=64, moe_top_k=2)
        self.dropout = dropout
        if residual is True:
            assert in_feats is not None
            self.res_connection = nn.Linear(in_feats, out_feats)
        else:
            self.res_connection = None

    def reset_parameters(self):
        """Reinitialize model parameters."""
        # self.graph_conv.reset_parameters()
        if self.res_connection is not None:
            self.res_connection.reset_parameters()

    def forward(self, graph, feats):
        new_feats = self.graph_conv(graph, feats)
        if self.res_connection is not None:
            res = self.res_connection
            new_feats = new_feats + res
            new_feats = F.dropout(new_feats, p=self.dropout, training=self.training)

        new_feats = self.pos_ff(new_feats)

        return new_feats


@register_model("moe_gcn")
class MoEGCN(BaseModel):
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
        parser.add_argument("--no-residual", action="store_true")
        parser.add_argument("--norm", type=str, default="batchnorm")
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
            not args.no_residual,
            args.norm,
        )

    def __init__(
        self, in_feats, hidden_size, out_feats, num_layers, dropout, activation="relu", residual=True, norm=None
    ):
        super(MoEGCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        conv_func = GraphConv
        conv_params = {
            "in_features": in_feats,
            "out_features": out_feats,
            "dropout": dropout,
            "norm": norm,
            "residual": residual,
            "activation": activation,
        }
        self.layers = nn.ModuleList(
            [
                GraphConvBlock(
                    conv_func,
                    conv_params,
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = get_activation(activation)

    def get_embeddings(self, graph):
        graph.sym_norm()

        h = graph.x
        for i in range(self.num_layers - 1):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.layers[i](graph, h)
        return h

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
            if i != self.num_layers - 1:
                h = self.act(h)
                h = F.dropout(h, self.dropout, training=self.training)
        return h

    def predict(self, data):
        return self.forward(data)
