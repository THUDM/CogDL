from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .gcn import GraphConvolution
from cogdl.data import Data

from cogdl.utils import get_activation, row_normalization, add_remaining_self_loops, dropout_adj
from torch_sparse import spspmm


class Pool(nn.Module):
    def __init__(
        self, in_feats: int, pooling_rate: float, aug_adj: bool = False, dropout: float = 0.5, activation: str = "tanh"
    ):
        super(Pool, self).__init__()
        self.aug_adj = aug_adj
        self.pooling_rate = pooling_rate
        self.act = get_activation(activation)
        self.proj = nn.Linear(in_feats, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[Data, torch.Tensor]:
        x = self.dropout(x)
        h = self.proj(x).squeeze()
        scores = self.act(h)
        return self.top_k(x, edge_index, scores)

    def top_k(self, x: torch.Tensor, edge_index: torch.Tensor, scores: torch.Tensor) -> Tuple[Data, torch.Tensor]:
        org_n_nodes = x.shape[0]
        num = int(self.pooling_rate * x.shape[0])
        values, indices = torch.topk(scores, max(2, num))

        if self.aug_adj:
            edge_attr = torch.ones(edge_index.shape[1])
            edge_index = edge_index.cpu()
            edge_index, _ = spspmm(edge_index, edge_attr, edge_index, edge_attr, org_n_nodes, org_n_nodes, org_n_nodes)
            edge_index = edge_index.to(x.device)

        batch = Data(x=x, edge_index=edge_index)
        new_batch = batch.subgraph(indices)
        num_nodes = new_batch.x.shape[0]
        new_batch.edge_attr = row_normalization(num_nodes, new_batch.edge_index)
        return new_batch, indices


class UnPool(nn.Module):
    def __init__(self):
        super(UnPool, self).__init__()

    def forward(self, num_nodes: int, h: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        new_h = torch.zeros(num_nodes, h.shape[1]).to(h.device)
        new_h[indices] = h
        return new_h


class GraphUnetLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        pooling_layer: int,
        pooling_rates: List[float],
        activation: str = "elu",
        dropout: float = 0.5,
        aug_adj: bool = False,
    ):
        super(GraphUnetLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.pooling_layer = pooling_layer
        self.gcn = GraphConvolution(hidden_size, hidden_size)
        self.act = get_activation(activation)

        self.down_gnns = nn.ModuleList([GraphConvolution(hidden_size, hidden_size) for _ in range(pooling_layer)])
        self.up_gnns = nn.ModuleList([GraphConvolution(hidden_size, hidden_size) for _ in range(pooling_layer)])
        self.poolings = nn.ModuleList(
            [Pool(hidden_size, pooling_rates[i], aug_adj, dropout) for i in range(pooling_layer)]
        )
        self.unpoolings = nn.ModuleList([UnPool() for _ in range(pooling_layer)])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]
    ) -> List[torch.Tensor]:
        adjs = []
        adj_attr = []
        indices = []
        down_hidden = []
        num_nodes = []
        h_list = []

        h_init = x
        h = x
        for i in range(self.pooling_layer):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.down_gnns[i](h, edge_index, edge_attr)
            h = self.act(h)

            adjs.append(edge_index)
            adj_attr.append(edge_attr)
            down_hidden.append(h)
            num_nodes.append(h.shape[0])

            s_g, index = self.poolings[i](h, edge_index)
            h, edge_index, edge_attr = s_g.x, s_g.edge_index, s_g.edge_attr
            indices.append(index)

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn(h, edge_index, edge_attr)
        h = self.act(h)

        for _i in range(self.pooling_layer):
            i = self.pooling_layer - _i - 1
            edge_index = adjs[i]
            edge_attr = adj_attr[i]
            index = indices[i]
            h = self.unpoolings[i](num_nodes[i], h, index)

            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.up_gnns[i](h, edge_index, edge_attr)
            h = self.act(h)

            h = h + down_hidden[i]
            h_list.append(h)
        h = h.add(h_init)
        h_list.append(h)
        return h_list


@register_model("unet")
class GraphUnet(BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--n-dropout", type=float, default=0.8)

        parser.add_argument("--adj-dropout", type=float, default=0.0)
        parser.add_argument("--n-pool", type=int, default=4)
        parser.add_argument("--pool-rate", nargs="+", default=[0.7, 0.5, 0.5, 0.4])
        parser.add_argument("--activation", type=str, default="relu")

        parser.add_argument("--improved", action="store_true")
        parser.add_argument("--aug-adj", action="store_true")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feats=args.num_features,
            hidden_size=args.hidden_size,
            out_feats=args.num_classes,
            pooling_layer=args.n_pool,
            pooling_rates=args.pool_rate,
            n_dropout=args.n_dropout,
            adj_dropout=args.adj_dropout,
            activation=args.activation,
            improved=args.improved,
            aug_adj=args.aug_adj,
        )

    def __init__(
        self,
        in_feats: int,
        hidden_size: int,
        out_feats: int,
        pooling_layer: int,
        pooling_rates: List[float],
        n_dropout: float = 0.5,
        adj_dropout: float = 0.3,
        activation: str = "elu",
        improved: bool = False,
        aug_adj: bool = False,
    ):
        super(GraphUnet, self).__init__()
        self.improved = improved
        self.n_dropout = n_dropout
        self.adj_dropout = adj_dropout

        self.act = get_activation(activation)
        assert pooling_layer <= len(pooling_rates)
        pooling_rates = pooling_rates[:pooling_layer]
        self.unet = GraphUnetLayer(hidden_size, pooling_layer, pooling_rates, activation, n_dropout, aug_adj)

        self.in_gcn = GraphConvolution(in_feats, hidden_size)
        self.out_gcn = GraphConvolution(hidden_size, out_feats)

        self.cache_edge_index = None
        self.cache_edge_attr = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.cache_edge_attr is None:
            edge_index, _ = add_remaining_self_loops(edge_index)
            if self.improved:
                self_loop = torch.stack([torch.arange(0, x.shape[0])] * 2, dim=0).to(x.device)
                edge_index = torch.cat([edge_index, self_loop], dim=1)

            edge_attr = row_normalization(x.shape[0], edge_index)
            self.cache_edge_attr = edge_attr
            self.cache_edge_index = edge_index
        else:
            edge_index = self.cache_edge_index
            edge_attr = self.cache_edge_attr

        if self.training and self.adj_dropout > 0:
            edge_index, edge_attr = dropout_adj(edge_index, edge_attr, self.adj_dropout)

        x = F.dropout(x, p=self.n_dropout, training=self.training)
        h = self.in_gcn(x, edge_index, edge_attr)
        h = self.act(h)
        h_list = self.unet(h, edge_index, edge_attr)

        h = h_list[-1]
        h = F.dropout(h, p=self.n_dropout, training=self.training)
        return self.out_gcn(h, edge_index, edge_attr)

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
