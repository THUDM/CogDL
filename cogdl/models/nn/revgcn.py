from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel
from .deepergcn import DeeperGCN
from .gat import GAT
from cogdl.layers.reversible_layer import RevGNNLayer
from cogdl.layers import GCNLayer, GATLayer, GENConv, ResGNNLayer
from cogdl.utils import get_activation, get_norm_layer, dropout_adj


def shared_dropout(x, dropout):
    m = torch.zeros_like(x).bernoulli_(1 - dropout)
    mask = m.requires_grad_(False) / (1 - dropout)
    return mask


class RevGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--group", type=int, default=2)
        parser.add_argument("--drop-edge-rate", type=float, default=0.0)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--norm", type=str, default="batchnorm")
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--num-layers", type=int, default=2)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.drop_edge_rate,
            args.activation,
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
        drop_edge_rate=0.1,
        activation="relu",
        norm="batchnorm",
        group=2,
    ):
        super(RevGCN, self).__init__()
        self.dropout = dropout
        self.drop_edge_rate = drop_edge_rate
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norm = get_norm_layer(norm, hidden_size)
        self.act = get_activation(activation)

        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNLayer(in_feats, hidden_size, residual=True,))
            elif i == num_layers - 1:
                self.layers.append(GCNLayer(hidden_size, out_feats, residual=True))
            else:
                conv = GCNLayer(hidden_size // group, hidden_size // group,)
                res_conv = ResGNNLayer(
                    conv,
                    hidden_size // group,
                    activation=activation,
                    norm=norm,
                    out_norm=norm,
                    out_channels=hidden_size // group,
                )
                self.layers.append(RevGNNLayer(res_conv, group))

    def forward(self, graph):
        graph.requires_grad = False
        edge_index, edge_weight = dropout_adj(
            graph.edge_index, drop_rate=self.drop_edge_rate, renorm=None, training=self.training
        )
        h = graph.x
        h = F.dropout(h, self.dropout, training=self.training)

        with graph.local_graph():
            graph.edge_index = edge_index
            graph.sym_norm()
            assert (graph.degrees() > 0).all()
            h = self.layers[0](graph, h)

            mask = shared_dropout(h, self.dropout)
            for i in range(1, len(self.layers) - 1):
                h = self.layers[i](graph, h, mask)
            h = self.norm(h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[-1](graph, h)

        return h


class RevGEN(BaseModel):
    @staticmethod
    def add_args(parser):
        DeeperGCN.add_args(parser)
        parser.add_argument("--group", type=int, default=2)
        parser.add_argument("--norm", type=str, default="batchnorm")
        parser.add_argument("--last-norm", type=str, default="batchnorm")
        parser.add_argument("--use-one-hot-emb", action="store_true")

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
            edge_attr_size=args.edge_attr_size,
            one_hot_emb=args.use_one_hot_emb if hasattr(args, "use_one_hot_emb") else False,
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
        edge_attr_size: Optional[list] = None,
        one_hot_emb: bool = False,
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
                edge_attr_size=edge_attr_size,
            )
            res_conv = ResGNNLayer(conv, hidden_size // group, norm=norm, activation=activation, residual=False)
            self.layers.append(RevGNNLayer(res_conv, group))
        self.activation = get_activation(activation)
        self.norm = get_norm_layer(last_norm, hidden_size)
        self.dropout = dropout
        if one_hot_emb:
            self.one_hot_encoder = nn.Linear(in_feats // 2, in_feats // 2)
        self.use_one_hot_emb = one_hot_emb

    def forward(self, graph):
        graph.requires_grad = False
        x = graph.x
        if self.use_one_hot_emb:
            x = x.split(2, dim=-1)
            x[1] = self.one_hot_encoder(x[1])
            x = torch.cat((x[0], x[1]), dim=1)
        h = self.input_fc(x)

        mask = shared_dropout(h, self.dropout)

        for layer in self.layers:
            h = layer(graph, h, mask)

        h = self.activation(self.norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.output_fc(h)
        return h


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
                self.layers.append(GATLayer(in_feats, hidden_size, nhead, alpha, attn_drop, residual=True,))
            elif i == num_layers - 1:
                self.layers.append(GATLayer(hidden_size * nhead, out_feats, 1, alpha, attn_drop, residual=True))
            else:
                conv = GATLayer(
                    hidden_size * nhead // group, hidden_size // group, nhead=nhead, alpha=alpha, attn_drop=attn_drop,
                )
                res_conv = ResGNNLayer(
                    conv,
                    hidden_size * nhead // group,
                    activation=activation,
                    norm=norm,
                    out_norm=norm,
                    out_channels=hidden_size * nhead // group,
                )
                self.layers.append(RevGNNLayer(res_conv, group))

    def forward(self, graph):
        graph.requires_grad = False
        h = graph.x
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.layers[0](graph, h)

        mask = shared_dropout(h, self.dropout)
        for i in range(1, len(self.layers) - 1):
            h = self.layers[i](graph, h, mask)
            if torch.isnan(h).any():
                print(f"! NaN - h_{i}")
                input()

        h = self.norm(h)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](graph, h)
        return h
