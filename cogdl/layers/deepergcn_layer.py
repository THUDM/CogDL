from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mlp_layer import MLP
from cogdl.utils import get_activation, edge_softmax, get_norm_layer, batch_max_pooling


class GENConv(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        aggr: str = "softmax_sg",
        beta: float = 1.0,
        p: float = 1.0,
        learn_beta: bool = False,
        learn_p: bool = False,
        use_msg_norm: bool = False,
        learn_msg_scale: bool = True,
        norm: Optional[str] = None,
        residual: bool = False,
        activation: Optional[str] = None,
        num_mlp_layers: int = 2,
        edge_attr_size: Optional[list] = None,
    ):
        super(GENConv, self).__init__()
        self.use_msg_norm = use_msg_norm
        self.mlp = MLP(in_feats, out_feats, in_feats * 2, num_layers=num_mlp_layers, activation=activation, norm=norm)

        self.message_encoder = torch.nn.ReLU()

        self.aggr = aggr
        if aggr == "softmax_sg":
            self.beta = torch.nn.Parameter(torch.Tensor([beta,]), requires_grad=learn_beta,)
        else:
            self.register_buffer("beta", None)
        if aggr == "powermean":
            self.p = torch.nn.Parameter(torch.Tensor([p,]), requires_grad=learn_p,)
        else:
            self.register_buffer("p", None)
        self.eps = 1e-7

        self.s = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=learn_msg_scale and use_msg_norm)
        self.residual = residual

        if edge_attr_size is not None and edge_attr_size[0] > 0:
            if len(edge_attr_size) > 1:
                self.edge_encoder = BondEncoder(edge_attr_size, in_feats)
            else:
                self.edge_encoder = EdgeEncoder(edge_attr_size[0], in_feats)
        else:
            self.edge_encoder = None

    def message_norm(self, x, msg):
        x_norm = torch.norm(x, dim=1, p=2)
        msg_norm = F.normalize(msg, p=2, dim=1)
        msg_norm = msg_norm * x_norm.unsqueeze(-1)
        return x + self.s * msg_norm

    def forward(self, graph, x):
        edge_index = graph.edge_index
        dim = x.shape[1]
        edge_msg = x[edge_index[1]]
        if self.edge_encoder is not None and graph.edge_attr is not None:
            edge_msg += self.edge_encoder(graph.edge_attr)
        edge_msg = self.message_encoder(edge_msg) + self.eps

        if self.aggr == "softmax_sg":
            h = edge_softmax(graph, self.beta * edge_msg.contiguous())
            h = edge_msg * h
        elif self.aggr == "softmax":
            h = edge_softmax(graph, edge_msg)
            h = edge_msg * h
        elif self.aggr == "powermean":
            deg = graph.degrees()
            torch.clamp_(edge_msg, 1e-7, 1.0)
            h = edge_msg.pow(self.p) / deg[edge_index[0]].unsqueeze(-1)
        elif self.aggr == "mean":
            deg = graph.degrees()
            deg_rev = deg.pow(-1)
            deg_rev[torch.isinf(deg_rev)] = 0
            h = edge_msg * deg_rev[edge_index[0]].unsqueeze(-1)
        else:
            h = edge_msg

        if self.aggr == "max":
            h = batch_max_pooling(h, edge_index[0])
        else:
            h = torch.zeros_like(x).scatter_add_(dim=0, index=edge_index[0].unsqueeze(-1).repeat(1, dim), src=h)
        if self.aggr == "powermean":
            h = h.pow(1.0 / self.p)
        if self.use_msg_norm:
            h = self.message_norm(x, h)

        if self.residual:
            h = h + x
        h = self.mlp(h)
        return h


class ResGNNLayer(nn.Module):
    """
    Implementation of DeeperGCN in paper `"DeeperGCN: All You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_

    Parameters
    -----------
    conv : nn.Module
        An instance of GNN Layer, recieving (graph, x) as inputs
    n_channels : int
        size of input features
    activation : str
    norm: str
        type of normalization, ``batchnorm`` as default
    dropout : float
    checkpoint_grad : bool
    """

    def __init__(
        self,
        conv,
        in_channels,
        activation="relu",
        norm="batchnorm",
        dropout=0.0,
        out_norm=None,
        out_channels=-1,
        residual=True,
        checkpoint_grad=False,
    ):
        super(ResGNNLayer, self).__init__()
        self.conv = conv
        self.activation = get_activation(activation)
        self.dropout = dropout
        self.norm = get_norm_layer(norm, in_channels)
        self.residual = residual
        if out_norm:
            self.out_norm = get_norm_layer(norm, out_channels)
        else:
            self.out_norm = None
        self.checkpoint_grad = checkpoint_grad

    def forward(self, graph, x, dropout=None, *args, **kwargs):
        h = self.norm(x)
        h = self.activation(h)
        if isinstance(dropout, float) or dropout is None:
            h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            if self.training:
                h = h * dropout

        if self.checkpoint_grad:
            h = checkpoint(self.conv, graph, h, *args, **kwargs)
        else:
            h = self.conv(graph, h, *args, **kwargs)
        if self.residual:
            h = h + x

        if self.out_norm:
            return self.out_norm(h)
        else:
            return h


class EdgeEncoder(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False):
        super(EdgeEncoder, self).__init__()
        self.nn = nn.Linear(in_feats, out_feats, bias=bias)

    def forward(self, edge_attr):
        return self.nn(edge_attr)


class BondEncoder(nn.Module):
    def __init__(self, bond_dim_list, emb_size):
        super(BondEncoder, self).__init__()
        self.bond_emb_list = nn.ModuleList()
        for i, size in enumerate(bond_dim_list):
            x = nn.Embedding(size, emb_size)
            self.bond_emb_list.append(x)

    def forward(self, edge_attr):
        out = 0
        for i in range(edge_attr.shape[1]):
            out += self.bond_emb_list[i](edge_attr[:, i])
        return out
