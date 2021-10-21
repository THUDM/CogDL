import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils.link_prediction_utils import GNNLinkPredict, sampling_edge_uniform
from cogdl.layers import RGCNLayer
from .. import BaseModel


class RGCN(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_layers,
        num_rels,
        regularizer="basis",
        num_bases=None,
        self_loop=True,
        dropout=0.0,
        self_dropout=0.0,
    ):
        super(RGCN, self).__init__()
        shapes = [in_feats] + [out_feats] * num_layers
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            RGCNLayer(shapes[i], shapes[i + 1], num_rels, regularizer, num_bases, self_loop, dropout, self_dropout)
            for i in range(num_layers)
        )

    def forward(self, graph, x):
        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](graph, h)
            if i < self.num_layers - 1:
                h = F.relu(h)
        return h


class LinkPredictRGCN(GNNLinkPredict, BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=200)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--regularizer", type=str, default="basis")
        parser.add_argument("--self-loop", action="store_false")
        parser.add_argument("--penalty", type=float, default=0.001)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--self-dropout", type=float, default=0.4)
        parser.add_argument("--num-bases", type=int, default=5)
        parser.add_argument("--sampling-rate", type=float, default=0.01)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            num_entities=args.num_entities,
            num_rels=args.num_rels,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            regularizer=args.regularizer,
            num_bases=args.num_bases,
            self_loop=args.self_loop,
            sampling_rate=args.sampling_rate,
            penalty=args.penalty,
            dropout=args.dropout,
            self_dropout=args.self_dropout,
        )

    def __init__(
        self,
        num_entities,
        num_rels,
        hidden_size,
        num_layers,
        regularizer="basis",
        num_bases=None,
        self_loop=True,
        sampling_rate=0.01,
        penalty=0,
        dropout=0.0,
        self_dropout=0.0,
    ):
        BaseModel.__init__(self)
        GNNLinkPredict.__init__(self)
        self.penalty = penalty
        self.num_nodes = num_entities
        self.num_rels = num_rels
        self.sampling_rate = sampling_rate
        self.edge_set = None

        self.model = RGCN(
            in_feats=hidden_size,
            out_feats=hidden_size,
            num_layers=num_layers,
            num_rels=num_rels,
            regularizer=regularizer,
            num_bases=num_bases,
            self_loop=self_loop,
            dropout=dropout,
            self_dropout=self_dropout,
        )
        # self.rel_weight = nn.Parameter(torch.Tensor(num_rels, hidden_size))
        # nn.init.xavier_normal_(self.rel_weight, gain=nn.init.calculate_gain("relu"))
        # self.emb = nn.Parameter(torch.Tensor(num_entities, hidden_size))
        # nn.init.xavier_normal_(self.emb, gain=nn.init.calculate_gain("relu"))

        self.rel_weight = nn.Embedding(num_rels, hidden_size)
        self.emb = nn.Embedding(num_entities, hidden_size)

    def forward(self, graph):
        reindexed_nodes, reindexed_edges = torch.unique(torch.stack(graph.edge_index), sorted=True, return_inverse=True)
        x = self.emb(reindexed_nodes)
        self.cahce_index = reindexed_nodes

        graph.edge_index = reindexed_edges
        # graph.num_nodes = reindexed_edges.max().item() + 1

        output = self.model(graph, x)
        # output = self.model(x, reindexed_indices, graph.edge_type)
        return output

    def loss(self, graph, scoring):
        edge_index = graph.edge_index
        edge_types = graph.edge_attr

        self.get_edge_set(edge_index, edge_types)
        batch_edges, batch_attr, samples, rels, labels = sampling_edge_uniform(
            edge_index, edge_types, self.edge_set, self.sampling_rate, self.num_rels
        )

        graph = graph.__class__(edge_index=batch_edges, edge_attr=batch_attr)
        # graph.edge_index = batch_edges
        # graph.edge_attr = batch_attr

        output = self.forward(graph)
        edge_weight = self.rel_weight(rels)
        sampled_nodes, reindexed_edges = torch.unique(samples, sorted=True, return_inverse=True)
        assert (sampled_nodes == self.cahce_index).any()
        sampled_types = torch.unique(rels)

        loss_n = self._loss(
            output[reindexed_edges[0]], output[reindexed_edges[1]], edge_weight, labels, scoring
        ) + self.penalty * self._regularization([self.emb(sampled_nodes), self.rel_weight(sampled_types)])
        return loss_n

    def predict(self, graph):
        device = next(self.parameters()).device
        indices = torch.arange(0, self.num_nodes).to(device)
        x = self.emb(indices)
        output = self.model(graph, x)

        return output, self.rel_weight.weight
