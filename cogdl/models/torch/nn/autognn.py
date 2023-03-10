import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GATLayer, SELayer, GCNLayer, GCNIILayer
from cogdl.models import BaseModel
from cogdl.utils import spmm


def gcn_model(in_feats, hidden_size, num_layers, out_feats, dropout, residual, norm, activation):
    shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]

    return nn.ModuleList(
        [
            GCNLayer(
                shapes[i],
                shapes[i + 1],
                dropout=dropout if i != num_layers - 1 else 0,
                residual=residual if i != num_layers - 1 else None,
                norm=norm if i != num_layers - 1 else None,
                activation=activation if i != num_layers - 1 else None,
            )
            for i in range(num_layers)
        ]
    )


def gat_model(
    in_feats, hidden_size, out_feats, nhead, attn_drop, alpha, residual, norm, num_layers, dropout, last_nhead
):
    layers = nn.ModuleList()
    layers.append(
        GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm)
    )
    if num_layers != 1:
        layers.append(nn.ELU())
    for i in range(num_layers - 2):
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(
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
        layers.append(nn.ELU())

    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(
        GATLayer(hidden_size * nhead, out_feats, attn_drop=attn_drop, alpha=alpha, nhead=last_nhead, residual=False,)
    )

    return layers


def grand_model(in_feats, hidden_size, out_feats, dropout, dropout2, norm):
    layers = nn.ModuleList()
    if norm == "batchnorm":
        layers.append(nn.BatchNorm1d(in_feats))
    layers.append(nn.Dropout(p=dropout))  # dropout=inputdropout
    layers.append(nn.Linear(in_feats, hidden_size))
    layers.append(nn.ReLU())
    if norm == "batchnorm":
        layers.append(nn.BatchNorm1d(hidden_size))
    layers.append(nn.Dropout(p=dropout2))  # dropout2
    layers.append(nn.Linear(hidden_size, out_feats))

    return layers


def gcnii_model(in_feats, hidden_size, out_feats, dropout, num_layers, alpha, lmbda, residual):
    layers = nn.ModuleList()
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_feats, hidden_size))
    layers.append(nn.ReLU())
    for i in range(num_layers):
        layers.append(nn.Dropout(p=dropout))
        layers.append(GCNIILayer(hidden_size, alpha, math.log(lmbda / (i + 1) + 1), residual))
        layers.append(nn.ReLU())
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_size, out_feats))

    return layers


def drgat_model(num_features, hidden_size, num_classes, dropout, num_heads):
    layers = nn.ModuleList()
    layers.append(nn.Dropout(p=dropout))
    layers.append(SELayer(num_features, se_channels=int(np.sqrt(num_features))))
    layers.append(GATLayer(num_features, hidden_size, nhead=num_heads, attn_drop=dropout))
    layers.append(nn.ELU())
    layers.append(nn.Dropout(p=dropout))
    layers.append(SELayer(hidden_size * num_heads, se_channels=int(np.sqrt(hidden_size * num_heads))))
    layers.append(GATLayer(hidden_size * num_heads, num_classes, nhead=1, attn_drop=dropout))
    layers.append(nn.ELU())

    return layers


class AutoGNN(BaseModel):
    """
    Args
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--layers-type", type=str, default="gcn")
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--attn-drop", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--last-nhead", type=int, default=1)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--dropoutn", type=float, default=0.5)

    @classmethod
    def build_model_from_args(cls, args):
        if not hasattr(args, "attn_drop"):
            args.attn_drop = 0.5
        if not hasattr(args, "alpha"):
            args.alpha = 0.2
        if not hasattr(args, "nhead"):
            args.nhead = 8
        if not hasattr(args, "last_nhead"):
            args.last_nhead = 1
        if not hasattr(args, "dropoutn"):
            args.dropoutn = 0.5
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.layers_type,
            args.dropout,
            args.activation,
            args.norm,
            args.residual,
            args.attn_drop,
            args.alpha,
            args.nhead,
            args.last_nhead,
            args.dropoutn,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        layers_type,
        dropout,
        activation=None,
        norm=None,  # reuse `use_bn`
        residual=False,
        attn_drop=0.5,  # reuse `dropnode`
        alpha=0.2,
        nhead=8,  # reuse `order`
        last_nhead=1,
        dropoutn=0.5,  # reuse `gcnii:lambda`
    ):
        super(AutoGNN, self).__init__()

        self.dropout = dropout
        self.layers_type = layers_type
        if self.layers_type == "gcn":
            self.layers = gcn_model(in_feats, hidden_size, num_layers, out_feats, dropout, residual, norm, activation)
            self.num_layers = num_layers

        elif self.layers_type == "gat":
            self.layers = gat_model(
                in_feats,
                hidden_size,
                out_feats,
                nhead,
                attn_drop,
                alpha,
                residual,
                norm,
                num_layers,
                dropout,
                last_nhead,
            )
            self.num_layers = num_layers
            self.last_nhead = last_nhead
        elif self.layers_type == "grand":
            self.layers = grand_model(in_feats, hidden_size, out_feats, dropout, dropoutn, norm)
            self.dropnode_rate = attn_drop
            self.order = nhead
        elif self.layers_type == "gcnii":
            self.layers = gcnii_model(in_feats, hidden_size, out_feats, dropout, num_layers, alpha, dropoutn, residual)
        elif self.layers_type == "drgat":
            self.layers = drgat_model(in_feats, hidden_size, out_feats, dropout, nhead)

        self.autognn_parameters = list(self.layers.parameters())

    def drop_node(self, x):
        n = x.shape[0]
        drop_rates = torch.ones(n) * self.dropnode_rate
        if self.training:
            masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
            x = masks.to(x.device) * x

        else:
            x = x * (1.0 - self.dropnode_rate)
        return x

    def rand_prop(self, graph, x):
        x = self.drop_node(x)

        y = x
        for i in range(self.order):
            x = spmm(graph, x).detach_()
            y.add_(x)
        return y.div_(self.order + 1.0).detach_()

    def normalize_x(self, x):
        row_sum = x.sum(1)
        row_inv = row_sum.pow_(-1)
        row_inv.masked_fill_(row_inv == float("inf"), 0)
        x = x * row_inv[:, None]
        return x

    def forward(self, graph):
        if self.layers_type == "gcn":
            graph.sym_norm()
            h = graph.x
        elif self.layers_type == "gat":
            h = graph.x
        elif self.layers_type == "grand":
            graph.sym_norm()
            x = graph.x
            x = self.normalize_x(x)
            h = self.rand_prop(graph, x)
        elif self.layers_type == "gcnii":
            graph.sym_norm()
            h = graph.x
        elif self.layers_type == "drgat":
            h = graph.x

        init_h = None
        for i, layer in enumerate(self.layers):

            if type(layer).__name__ == "GATLayer" or type(layer).__name__ == "GCNLayer":
                h = layer(graph, h)
            elif type(layer).__name__ == "GCNIILayer":
                h = layer(graph, h, init_h)
            else:
                h = layer(h)

            if i == 2:
                init_h = h
        return h
