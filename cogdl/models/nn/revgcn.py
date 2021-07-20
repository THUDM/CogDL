import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .deepergcn import DeeperGCN
from .gat import GAT
from .gcn import TKipfGCN
from cogdl.layers.reversible_layer import RevGNNLayer
from cogdl.layers import GCNLayer, GATLayer, GENConv, ResGNNLayer
from cogdl.utils import get_activation, get_norm_layer


def shared_dropout(x, dropout):
    m = torch.zeros_like(x).bernoulli_(1 - dropout)
    mask = m.requires_grad_(False) / (1 - dropout)
    return mask


@register_model("revgcn")
class RevGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--group", type=int, default=2)
        TKipfGCN.add_args(parser)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.activation,
            args.residual,
            args.norm,
            args.group,
        )

    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        dropout=0.5,
        activation="relu",
        residual=False,
        norm=None,
        group=2,
    ):
        super(RevGCN, self).__init__()
        self.input_fc = nn.Linear(in_feats, hidden_size)
        self.output_fc = nn.Linear(hidden_size, out_feats)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = GCNLayer(
                hidden_size // group,
                hidden_size // group,
            )
            res_conv = ResGNNLayer(conv, hidden_size // group, norm=norm, activation=activation)
            self.layers.append(RevGNNLayer(res_conv, group))
        self.activation = get_activation(activation)
        self.norm = get_norm_layer(norm, hidden_size)
        self.dropout = dropout

    def forward(self, graph):
        graph.requires_grad = False
        h = self.input_fc(graph.x)

        mask = shared_dropout(h, self.dropout)

        for layer in self.layers:
            h = layer(graph, h, mask)

        h = self.activation(self.norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.output_fc(h)
        return h


@register_model("revgen")
class RevGEN(BaseModel):
    @staticmethod
    def add_args(parser):
        DeeperGCN.add_args(parser)
        parser.add_argument("--group", type=int, default=2)
        parser.add_argument("--norm", type=str, default="batchnorm")
        parser.add_argument("--last-norm", type=str, default="batchnorm")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.group,
            args.activation,
            args.norm,
            args.last_norm,
            args.dropout,
            args.aggr,
            args.beta,
            args.p,
            args.learn_beta,
            args.learn_p,
            args.learn_msg_scale,
            args.use_msg_norm,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        group=2,
        activation="relu",
        norm="batchnorm",
        last_norm="batchnorm",
        dropout=0.0,
        aggr="softmax_sg",
        beta=1.0,
        p=1.0,
        learn_beta=False,
        learn_p=False,
        learn_msg_scale=True,
        use_msg_norm=False,
    ):
        super(RevGEN, self).__init__()
        self.input_fc = nn.Linear(in_feats, hidden_size)
        self.output_fc = nn.Linear(hidden_size, out_feats)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = GENConv(
                hidden_size // group,
                hidden_size // group,
                aggr,
                beta,
                p,
                learn_beta,
                learn_p,
                use_msg_norm,
                learn_msg_scale,
                residual=True,
            )
            res_conv = ResGNNLayer(conv, hidden_size // group, norm=norm, activation=activation)
            self.layers.append(RevGNNLayer(res_conv, group))
        self.activation = get_activation(activation)
        self.norm = get_norm_layer(last_norm, hidden_size)
        self.dropout = dropout

    def forward(self, graph):
        graph.requires_grad = False
        x = graph.x
        h = self.input_fc(x)

        mask = shared_dropout(h, self.dropout)

        for layer in self.layers:
            h = layer(graph, h, mask)

        h = self.activation(self.norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.output_fc(h)
        return h


@register_model("revgat")
class RevGAT(BaseModel):
    @staticmethod
    def add_args(parser):
        GAT.add_args(parser)
        parser.add_argument("--norm", type=str, default="batchnorm")
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--group", type=int, default=2)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.group,
            args.alpha,
            args.nhead,
            args.dropout,
            args.attn_drop,
            args.activation,
            args.norm,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        group=2,
        alpha=0.2,
        nhead=1,
        dropout=0.5,
        attn_drop=0.5,
        activation="relu",
        norm="batchnorm",
    ):
        super(RevGAT, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norm = get_norm_layer(norm, hidden_size * nhead)
        self.act = get_activation(activation)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    GATLayer(
                        in_feats,
                        hidden_size,
                        nhead,
                        alpha,
                        attn_drop,
                        residual=True,
                    )
                )
            elif i == num_layers - 1:
                self.layers.append(GATLayer(hidden_size * nhead, out_feats, 1, alpha, attn_drop, residual=True))
            else:
                conv = GATLayer(
                    hidden_size * nhead // group,
                    hidden_size // group,
                    nhead=nhead,
                    alpha=alpha,
                    attn_drop=attn_drop,
                    residual=True,
                    activation=activation,
                    norm=norm,
                )
                res_conv = ResGNNLayer(conv, hidden_size * nhead // group, activation=activation, norm=norm)
                self.layers.append(RevGNNLayer(res_conv, group))

    def forward(self, graph):
        graph.requires_grad = False
        h = graph.x
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.layers[0](graph, h)

        mask = shared_dropout(h, self.dropout)
        for i in range(1, len(self.layers) - 1):
            h = self.layers[i](graph, h, mask)

        h = self.norm(h)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](graph, h)
        return h
