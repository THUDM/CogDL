import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .deepergcn import DeeperGCN
from .gat import GAT
from cogdl.layers.reversible_layer import RevGNNLayer
from cogdl.layers.deepergcn_layer import GENConv
from cogdl.layers.gat_layer import GATLayer
from cogdl.utils import get_activation, get_norm_layer


def shared_dropout(x, mask, training):
    if training:
        return x * mask
    else:
        return x


@register_model("revgcn")
class RevGEN(BaseModel):
    @staticmethod
    def add_args(parser):
        DeeperGCN.add_args(parser)
        parser.add_argument("--group", type=int, default=2)
        parser.add_argument("--norm", type=str, default="batchnorm")

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
        in_feat,
        hidden_size,
        out_feat,
        num_layers,
        group=2,
        activation="relu",
        norm="batchnorm",
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
        self.input_fc = nn.Linear(in_feat, hidden_size)
        self.output_fc = nn.Linear(hidden_size, out_feat)
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
            )
            self.layers.append(RevGNNLayer(conv, group))
        self.activation = get_activation(activation)
        self.norm = get_norm_layer(norm, hidden_size)
        self.dropout = dropout

    def forward(self, graph):
        graph.requires_grad = False
        x = graph.x
        h = self.input_fc(x)

        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)

        for layer in self.layers:
            h = shared_dropout(h, mask, self.training)
            h = layer(graph, h)

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
            args.last_nhead,
            args.dropout,
            args.attn_drop,
            args.activation,
            args.norm,
            args.residual,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feat,
        num_layers,
        group=2,
        alpha=0.2,
        nhead=1,
        last_nhead=1,
        dropout=0.5,
        attn_drop=0.5,
        activation="relu",
        norm="batchnorm",
        residual=True,
    ):
        super(RevGAT, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    GATLayer(
                        in_feats, hidden_size, nhead, alpha, attn_drop, residual=True, activation=activation, norm=norm
                    )
                )
            elif i == num_layers - 1:
                self.layers.append(GATLayer(hidden_size, out_feat, last_nhead, alpha, attn_drop, residual=residual))
            else:
                conv = GATLayer(
                    hidden_size,
                    hidden_size,
                    nhead=nhead,
                    alpha=alpha,
                    attn_drop=attn_drop,
                    residual=residual,
                    activation=activation,
                    norm=norm,
                )
                self.layers.append(RevGNNLayer(conv, group))

    def forward(self, graph):
        graph.requires_grad = False
        h = graph.x
        h = F.dropout(h, self.dropout, training=self.training)
        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)

        for i, layer in enumerate(self.layers):
            h = self.layers[i](graph, h)
            if i < self.num_layers - 2:
                h = shared_dropout(h, mask, training=self.training)
            elif i == self.num_layers - 2:
                h = F.dropout(h, self.dropout, training=self.training)
        return h
