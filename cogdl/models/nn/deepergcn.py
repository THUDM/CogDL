from typing import Any
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import register_model, BaseModel
from cogdl.utils import mul_edge_softmax, get_activation
from cogdl.trainers.sampled_trainer import RandomClusterTrainer


class GENConv(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        aggr="softmax_sg",
        beta=1.0,
        p=1.0,
        learn_beta=False,
        learn_p=False,
        use_msg_norm=False,
        learn_msg_scale=True,
    ):
        super(GENConv, self).__init__()
        self.use_msg_norm = use_msg_norm
        self.mlp = nn.Linear(in_feat, out_feat)

        self.message_encoder = torch.nn.ReLU()

        self.aggr = aggr
        if aggr == "softmax_sg":
            self.beta = torch.nn.Parameter(
                torch.Tensor(
                    [
                        beta,
                    ]
                ),
                requires_grad=learn_beta,
            )
        else:
            self.register_buffer("beta", None)
        if aggr == "powermean":
            self.p = torch.nn.Parameter(
                torch.Tensor(
                    [
                        p,
                    ]
                ),
                requires_grad=learn_p,
            )
        else:
            self.register_buffer("p", None)
        self.eps = 1e-7

        self.s = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=learn_msg_scale)
        self.act = nn.ReLU()

    def message_norm(self, x, msg):
        x_norm = torch.norm(x, dim=1, p=2)
        msg_norm = F.normalize(msg, p=2, dim=1)
        msg_norm = msg_norm * x_norm.unsqueeze(-1)
        return x + self.s * msg_norm

    def forward(self, graph, x):
        edge_index = graph.edge_index
        dim = x.shape[1]
        edge_msg = x[edge_index[1]]  # if edge_attr is None else x[edge_index[1]] + edge_attr
        edge_msg = self.act(edge_msg) + self.eps

        if self.aggr == "softmax_sg":
            h = mul_edge_softmax(graph, self.beta * edge_msg)
            h = edge_msg * h
        elif self.aggr == "softmax":
            h = mul_edge_softmax(graph, edge_msg)
            h = edge_msg * h
        elif self.aggr == "powermean":
            deg = graph.degrees()
            h = edge_msg.pow(self.t) / deg[edge_index[0]].unsqueeze(-1)
        else:
            raise NotImplementedError

        h = torch.zeros_like(x).scatter_add_(dim=0, index=edge_index[0].unsqueeze(-1).repeat(1, dim), src=h)
        if self.aggr == "powermean":
            h = h.pow(1.0 / self.p)
        if self.use_msg_norm:
            h = self.message_norm(x, h)
        h = self.mlp(h)
        return h


class DeepGCNLayer(nn.Module):
    """
    Implementation of DeeperGCN in paper `"DeeperGCN: All You Need to Train Deeper GCNs"` <https://arxiv.org/abs/2006.07739>

    Parameters
    -----------
    in_feat : int
        Size of each input sample
    out_feat : int
        Size of each output sample
    conv : class
        Base convolution layer.
    connection : str
        Residual connection type, `res` or `res+`.
    activation : str
    dropout : float
    checkpoint_grad : bool
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        conv,
        connection="res",
        activation="relu",
        dropout=0.0,
        checkpoint_grad=False,
    ):
        super(DeepGCNLayer, self).__init__()
        self.conv = conv
        self.activation = get_activation(activation)
        self.dropout = dropout
        self.connection = connection
        self.norm = nn.BatchNorm1d(out_feat, affine=True)
        self.checkpoint_grad = checkpoint_grad

    def forward(self, graph, x):
        if self.connection == "res+":
            h = self.norm(x)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.checkpoint_grad:
                h = checkpoint(self.conv, graph, h)
            else:
                h = self.conv(graph, h)
        elif self.connection == "res":
            h = self.conv(graph, x)
            h = self.norm(h)
            h = self.activation(h)
        else:
            raise NotImplementedError
        return x + h


@register_model("deepergcn")
class DeeperGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=14)
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--connection", type=str, default="res+")
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--aggr", type=str, default="softmax_sg")
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--p", type=float, default=1.0)
        parser.add_argument("--learn-beta", action="store_true")
        parser.add_argument("--learn-p", action="store_true")
        parser.add_argument("--learn-msg-scale", action="store_true")
        parser.add_argument("--use-msg-norm", action="store_true")
        # fmt: on

        """
            ogbn-products:
                num_layers: 14
                self_loop:
                aggr: softmax_sg
                beta: 0.1
        """

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feat=args.num_features,
            hidden_size=args.hidden_size,
            out_feat=args.num_classes,
            num_layers=args.num_layers,
            connection=args.connection,
            activation=args.connection,
            dropout=args.dropout,
            aggr=args.aggr,
            beta=args.beta,
            p=args.p,
            learn_beta=args.learn_beta,
            learn_p=args.learn_p,
            learn_msg_scale=args.learn_msg_scale,
            use_msg_norm=args.use_msg_norm,
        )

    def __init__(
        self,
        in_feat,
        hidden_size,
        out_feat,
        num_layers,
        connection="res+",
        activation="relu",
        dropout=0.0,
        aggr="max",
        beta=1.0,
        p=1.0,
        learn_beta=False,
        learn_p=False,
        learn_msg_scale=True,
        use_msg_norm=False,
    ):
        super(DeeperGCN, self).__init__()
        self.dropout = dropout
        self.feat_encoder = nn.Linear(in_feat, hidden_size)

        self.layers = nn.ModuleList()
        self.layers.append(GENConv(hidden_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(
                DeepGCNLayer(
                    in_feat=hidden_size,
                    out_feat=hidden_size,
                    conv=GENConv(
                        in_feat=hidden_size,
                        out_feat=hidden_size,
                        aggr=aggr,
                        beta=beta,
                        p=p,
                        learn_beta=learn_beta,
                        learn_p=learn_p,
                        use_msg_norm=use_msg_norm,
                        learn_msg_scale=learn_msg_scale,
                    ),
                    connection=connection,
                    activation=activation,
                    dropout=dropout,
                    checkpoint_grad=(num_layers > 3) and ((i + 1) == num_layers // 2),
                )
            )
        self.norm = nn.BatchNorm1d(hidden_size, affine=True)
        self.activation = get_activation(activation)
        self.fc = nn.Linear(hidden_size, out_feat)

    def forward(self, graph):
        x = graph.x
        h = self.feat_encoder(x)
        for layer in self.layers:
            h = layer(graph, h)
        h = self.activation(self.norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc(h)
        return h

    def predict(self, graph):
        return self.forward(graph)

    @staticmethod
    def get_trainer(taskType: Any, args):
        return RandomClusterTrainer
