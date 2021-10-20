from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.data import Graph
from cogdl.layers import GCNLayer
from cogdl.utils import dropout_adj, get_activation
from torch_sparse import spspmm

from .. import BaseModel


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

    def forward(self, graph: Graph, x: torch.Tensor) -> Tuple[Graph, torch.Tensor]:
        x = self.dropout(x)
        h = self.proj(x).squeeze()
        scores = self.act(h)
        return self.top_k(graph, x, scores)

    def top_k(self, graph, x: torch.Tensor, scores: torch.Tensor) -> Tuple[Graph, torch.Tensor]:
        org_n_nodes = x.shape[0]
        num = int(self.pooling_rate * x.shape[0])
        values, indices = torch.topk(scores, max(2, num))

        if self.aug_adj:
            edge_attr = torch.ones(graph.edge_index[0].shape[0])
            edge_index = torch.stack(graph.edge_index).cpu()
            edge_index, _ = spspmm(edge_index, edge_attr, edge_index, edge_attr, org_n_nodes, org_n_nodes, org_n_nodes)
            edge_index = edge_index.to(x.device)
            batch = Graph(x=x, edge_index=edge_index)
        else:
            batch = graph
            batch.x = x
        new_batch = batch.subgraph(indices)

        new_batch.row_norm()
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
        self.gcn = GCNLayer(hidden_size, hidden_size)
        self.act = get_activation(activation)

        self.down_gnns = nn.ModuleList([GCNLayer(hidden_size, hidden_size) for _ in range(pooling_layer)])
        self.up_gnns = nn.ModuleList([GCNLayer(hidden_size, hidden_size) for _ in range(pooling_layer)])
        self.poolings = nn.ModuleList(
            [Pool(hidden_size, pooling_rates[i], aug_adj, dropout) for i in range(pooling_layer)]
        )
        self.unpoolings = nn.ModuleList([UnPool() for _ in range(pooling_layer)])

    def forward(self, graph: Graph, x: torch.Tensor) -> List[torch.Tensor]:
        adjs = []
        indices = []
        down_hidden = []
        num_nodes = []
        h_list = []

        h_init = x
        h = x

        g = graph
        for i in range(self.pooling_layer):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.down_gnns[i](g, h)
            h = self.act(h)

            adjs.append(g)
            down_hidden.append(h)
            num_nodes.append(h.shape[0])

            g, index = self.poolings[i](g, h)
            h = g.x
            indices.append(index)

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn(g, h)
        h = self.act(h)

        for _i in range(self.pooling_layer):
            i = self.pooling_layer - _i - 1
            g = adjs[i]
            index = indices[i]
            h = self.unpoolings[i](num_nodes[i], h, index)

            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.up_gnns[i](g, h)
            h = self.act(h)

            h = h + down_hidden[i]
            h_list.append(h)
        h = h.add(h_init)
        h_list.append(h)
        return h_list


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
        pooling_rates = [float(x) for x in pooling_rates]
        self.unet = GraphUnetLayer(hidden_size, pooling_layer, pooling_rates, activation, n_dropout, aug_adj)

        self.in_gcn = GCNLayer(in_feats, hidden_size)
        self.out_gcn = GCNLayer(hidden_size, out_feats)

    def forward(self, graph: Graph) -> torch.Tensor:
        x = graph.x
        if self.improved and not hasattr(graph, "unet_improved"):
            row, col = graph.edge_index
            row = torch.cat([row, torch.arange(0, x.shape[0], device=x.device)], dim=0)
            col = torch.cat([col, torch.arange(0, x.shape[0], device=x.device)], dim=0)
            graph.edge_index = (row, col)
            graph["unet_improved"] = True
        graph.row_norm()

        with graph.local_graph():
            if self.training and self.adj_dropout > 0:
                graph.edge_index, graph.edge_weight = dropout_adj(graph.edge_index, graph.edge_weight, self.adj_dropout)

            x = F.dropout(x, p=self.n_dropout, training=self.training)
            h = self.in_gcn(graph, x)
            h = self.act(h)
            h_list = self.unet(graph, h)

            h = h_list[-1]
            h = F.dropout(h, p=self.n_dropout, training=self.training)
            return self.out_gcn(graph, h)
