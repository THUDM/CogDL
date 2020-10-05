import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import row_normalization, spmm
from cogdl.layers.link_prediction_module import GNNLinkPredict, cal_mrr, sampling_edge_uniform
from .. import register_model, BaseModel


class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_edge_types, regularizer="basis", num_bases=None, self_loop=True,
                 dropout=0.0, self_dropout=0.0, layer_norm=True, bias=True):
        super(RGCNLayer, self).__init__()
        self.num_bases = num_bases
        self.regularizer = regularizer
        self.num_edge_types = num_edge_types
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.self_loop = self_loop
        self.dropout = dropout
        self.self_dropout = self_dropout

        if self.num_bases is None or self.num_bases > num_edge_types or self.num_bases < 0:
            self.num_bases = num_edge_types

        if regularizer == "basis":
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, in_feats, out_feats))
            if self.num_bases < num_edge_types:
                self.alpha = nn.Parameter(torch.Tensor(num_edge_types, self.num_bases))
            else:
                self.register_buffer("alpha", None)
        elif regularizer == "bdd":
            assert (in_feats % num_bases == 0) and (out_feats % num_bases == 0)
            self.block_in_feats = in_feats // num_bases
            self.block_out_feats = out_feats // num_bases
            self.weight = nn.Parameter(
                torch.Tensor(num_edge_types, self.num_bases, self.block_in_feats * self.block_out_feats))
        else:
            raise NotImplementedError

        if bias is True:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)

        if self_loop:
            self.weight_self_loop = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_buffer("weight_self_loop", None)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.register_buffer("layer_norm", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        if self.alpha is not None:
            nn.init.xavier_uniform_(self.alpha, gain=nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.self_loop is not None:
            nn.init.xavier_uniform_(self.weight_self_loop, gain=nn.init.calculate_gain("relu"))

    def forward(self, x, edge_index, edge_type):
        if self.regularizer == "basis":
            h_list = self.basis_forward(x, edge_index, edge_type)
        else:
            h_list = self.bdd_forward(x, edge_index, edge_type)

        h_result = sum(h_list)
        h_result = F.dropout(h_result, p=self.dropout, training=self.training)
        if self.layer_norm is not None:
            h_result = self.layer_norm(h_result)
        if self.bias is not None:
            h_result = h_result + self.bias
        if self.self_loop is not None:
            h_result += F.dropout(torch.matmul(x, self.weight_self_loop), p=self.self_dropout, training=self.training)
        return h_result

    def basis_forward(self, x, edge_index, edge_type):
        if self.num_bases < self.num_edge_types:
            weight = torch.matmul(self.alpha, self.weight.view(self.num_bases, -1))
            weight = weight.view(self.num_edge_types, self.in_feats, self.out_feats)
        else:
            weight = self.weight
        edge_weight = torch.ones(edge_type.shape).to(x.device)
        edge_weight = row_normalization(x.shape[0], edge_index, edge_weight)

        h = torch.matmul(x, weight)  # (N, d1) by (r, d1, d2) -> (r, N, d2)

        h_list = []
        for edge_t in range(self.num_edge_types):
            edge_mask = (edge_type == edge_t)
            _edge_index_t = edge_index.t()[edge_mask].t()
            temp = spmm(_edge_index_t, edge_weight[edge_mask], h[edge_t])
            h_list.append(temp)
        return h_list

    def bdd_forward(self, x, edge_index, edge_type):
        _x = x.view(-1, self.num_bases, self.block_in_feats)

        edge_weight = torch.ones(edge_type.shape).to(x.device)
        edge_weight = row_normalization(x.shape[0], edge_index, edge_weight)

        h_list = []
        for edge_t in range(self.num_edge_types):
            _weight = self.weight[edge_t].view(self.num_bases, self.block_in_feats, self.block_out_feats)
            edge_mask = (edge_type == edge_t)
            _edge_index_t = edge_index.t()[edge_mask].t()
            h_t = torch.einsum("abc,bcd->abd", _x, _weight).reshape(-1, self.out_feats)
            h_t = spmm(_edge_index_t, edge_weight[edge_mask], h_t)
            h_list.append(h_t)

        return h_list


class RGCN(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers, num_rels, regularizer="basis", num_bases=None, self_loop=True,
                 dropout=0.0, self_dropout=0.0):
        super(RGCN, self).__init__()
        shapes = [in_feats] + [out_feats] * num_layers
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            RGCNLayer(shapes[i], shapes[i + 1], num_rels, regularizer, num_bases, self_loop, dropout, self_dropout)
            for i in range(num_layers)
        )

    def forward(self, x, edge_index, edge_type):
        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](x, edge_index, edge_type)
            if i < self.num_layers - 1:
                h = F.relu(h)
        return h


@register_model("rgcn")
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
            self_dropout=args.self_dropout
        )

    def __init__(self, num_entities, num_rels, hidden_size, num_layers, regularizer="basis", num_bases=None,
                 self_loop=True, sampling_rate=0.01, penalty=0, dropout=0.0, self_dropout=0.0):
        BaseModel.__init__(self)
        GNNLinkPredict.__init__(self, "distmult", hidden_size)
        self.penalty = penalty
        self.num_nodes = num_entities
        self.num_rels = num_rels
        self.sampling_rate = sampling_rate
        self.edge_set = None

        self.cahce_index = None

        self.model = RGCN(
            in_feats=hidden_size,
            out_feats=hidden_size,
            num_layers=num_layers,
            num_rels=num_rels,
            regularizer=regularizer,
            num_bases=num_bases,
            self_loop=self_loop,
            dropout=dropout,
            self_dropout=self_dropout,)
        # self.rel_weight = nn.Parameter(torch.Tensor(num_rels, hidden_size))
        # nn.init.xavier_normal_(self.rel_weight, gain=nn.init.calculate_gain("relu"))
        # self.emb = nn.Parameter(torch.Tensor(num_entities, hidden_size))
        # nn.init.xavier_normal_(self.emb, gain=nn.init.calculate_gain("relu"))

        self.rel_weight = nn.Embedding(num_rels, hidden_size)
        self.emb = nn.Embedding(num_entities, hidden_size)

    def forward(self, edge_index, edge_type):
        reindexed_nodes, reindexed_indices = torch.unique(edge_index, sorted=True, return_inverse=True)
        x = self.emb(reindexed_nodes)
        self.cahce_index = reindexed_nodes
        output = self.model(x, reindexed_indices, edge_type)
        return output

    def loss(self, data, split="train"):
        if split == "train":
            mask = data.train_mask
        elif split == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask
        edge_index, edge_types = data.edge_index[:, mask], data.edge_attr[mask]

        self.get_edge_set(edge_index, edge_types)
        batch_edges, batch_attr, samples, rels, labels = sampling_edge_uniform(edge_index, edge_types, self.edge_set, self.sampling_rate, self.num_rels)
        output = self.forward(batch_edges, batch_attr)
        edge_weight = self.rel_weight(rels)
        sampled_nodes, reindexed_edges = torch.unique(samples, sorted=True, return_inverse=True)
        assert (sampled_nodes == self.cahce_index).any()
        sampled_types = torch.unique(rels)

        loss_n = self._loss(output[reindexed_edges[0]], output[reindexed_edges[1]], edge_weight, labels) \
                 + self.penalty * self._regularization([self.emb(sampled_nodes), self.rel_weight(sampled_types)])
        return loss_n

    def predict(self, edge_index, edge_type):
        indices = torch.arange(0, self.num_nodes).to(edge_index.device)
        x = self.emb(indices)
        output = self.model(x, edge_index, edge_type)
        mrr, hits = cal_mrr(output, self.rel_weight.weight, edge_index, edge_type, scoring=self.scoring, protocol="raw", batch_size=500, hits=[1, 3, 10])
        return mrr, hits
