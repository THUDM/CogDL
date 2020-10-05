from ast import parse
from cogdl import data
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import row_normalization
from cogdl.layers.link_prediction_module import GNNLinkPredict, sampling_edge_uniform, cal_mrr 
from .. import BaseModel, register_model


def com_mult(a, b):
    """Borrowed from https://github.com/malllabiisc/CompGCN"""
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
    """Borrowed from https://github.com/malllabiisc/CompGCN"""
    a[..., 1] = -a[..., 1]
    return a

def ccorr(a, b):
    """Borrowed from https://github.com/malllabiisc/CompGCN"""
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class BasesRelEmbLayer(nn.Module):
    def __init__(self, num_bases, num_rels, in_feats):
        super(BasesRelEmbLayer, self).__init__()
        self.num_bases = num_bases
        self.num_resl = num_rels
        self.in_feats = in_feats
        self.weight = nn.Parameter(torch.Tensor(num_bases, in_feats))
        self.alpha = nn.Parameter(torch.Tensor(2*num_rels, num_bases))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.alpha)
    
    def forward(self):
        weight = torch.matmul(self.alpha, self.weight)
        return weight


class CompGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels, opn="mult", num_bases=None, activation=lambda x:x, dropout=0.0, bias=True):
        super(CompGCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_rels = num_rels
        self.opn = opn
        self.use_bases = (num_bases is not None and num_bases > 0)

        self.weight_in = self.get_param(in_feats, out_feats)
        self.weight_out = self.get_param(in_feats, out_feats)
        self.weight_rel = self.get_param(in_feats, out_feats)
        self.weight_loop = self.get_param(in_feats, out_feats)
        self.loop_rel = self.get_param(1, in_feats)

        if self.use_bases:
            self.basis_weight = BasesRelEmbLayer(num_bases, num_rels, in_feats)
        else:
            self.register_buffer("basis_weight", None)

        self.dropout = dropout
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.register_buffer("bias", None)
        self.bn = nn.BatchNorm1d(out_feats)

    def get_param(self, num_in, num_out):
        weight = nn.Parameter(torch.Tensor(num_in, num_out))
        nn.init.xavier_normal_(weight.data)
        return weight

    def forward(self, x, edge_index, edge_type, rel_embed=None):
        device = x.device
        if self.use_bases:
            rel_embed = self.basis_weight()
        rel_embed = torch.cat((rel_embed, self.loop_rel), dim=0)
        num_edges = edge_index.shape[1]//2
        num_entities = x.shape[0]

        index, rev_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        loop_index = torch.stack((torch.arange(num_entities), torch.arange(num_entities))).to(device)
        types, rev_types = edge_type[:num_edges], edge_type[num_edges:]
        loop_types = torch.full((num_entities,), rel_embed.shape[0]-1, dtype=torch.long).to(device)

        in_norm = row_normalization(num_entities, index)
        rev_norm = row_normalization(num_entities, rev_index)

        emb = self.message_passing(x, rel_embed, index, types, "in", in_norm)
        rev_emb = self.message_passing(x, rel_embed, rev_index, rev_types, "out", rev_norm)
        loop_emb = self.message_passing(x, rel_embed, loop_index, loop_types, "loop")

        out = 1/3 * (emb + rev_emb + loop_emb)
        if self.bias is not None:
            out += self.bias
        out = self.bn(out)
        return self.activation(out), torch.matmul(rel_embed, self.weight_rel)[:-1]

    def message_passing(self, x, rel_embed, edge_index, edge_types, mode, edge_weight=None):
        device = x.device
        tail_emb = x[edge_index[1]]
        rel_emb = rel_embed[edge_types]
        weight = getattr(self, f"weight_{mode}")

        trans_embed = self.rel_transform(tail_emb, rel_emb)
        trans_embed = torch.matmul(trans_embed, weight)
        dim = trans_embed.shape[1]
        if edge_weight is not None:
            trans_embed = trans_embed * edge_weight.unsqueeze(-1)
        embed = torch.zeros(x.shape[0], dim).to(device).scatter_add_(0, edge_index[0].unsqueeze(-1).repeat(1, dim), trans_embed)
        return F.dropout(embed, p=self.dropout, training=self.training)

    def rel_transform(self, ent_embed, rel_embed):
        if self.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed


class CompGCN(nn.Module):
    def __init__(self, num_entities, num_rels, num_bases, in_feats, hidden_size, out_feats, layers, dropout, activation):
        super(CompGCN, self).__init__()
        self.opn = "corr"
        self.num_rels = num_rels
        self.num_entities = num_entities
        if num_bases is not None and num_bases > 0:
            self.init_rel = nn.Embedding(num_bases, in_feats)
        else:
            self.init_rel = nn.Embedding(2*num_rels, in_feats)

        self.convs = nn.ModuleList()
        if num_bases > 0:
            self.convs.append(CompGCNLayer(in_feats=in_feats, out_feats=hidden_size, num_rels=num_rels,
                                          opn=self.opn, num_bases=num_bases, activation=activation, dropout=dropout))
        else:
            self.convs.append(CompGCNLayer(
                in_feats=in_feats,
                out_feats=hidden_size,
                num_rels=num_rels,
                opn=self.opn,
                activation=activation,
                dropout=dropout
            ))
        if layers == 2:
            self.convs.append(CompGCNLayer(
                in_feats=hidden_size,
                out_feats=out_feats,
                num_rels=num_rels,
                opn=self.opn,
                activation=activation,
                dropout=dropout))

    def forward(self, x, edge_index, edge_types):
        rel_embed = self.init_rel.weight
        node_embed = x
        for layer in self.convs:
            node_embed, rel_embed = layer(node_embed, edge_index, edge_types, rel_embed)
        return node_embed, rel_embed


@register_model("compgcn")
class LinkPredictCompGCN(GNNLinkPredict, BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=200)
        parser.add_argument("--penalty", type=float, default=0.001)
        parser.add_argument("--dropout", type=float, default=0.3)
        parser.add_argument("--num-bases", type=int, default=10)
        parser.add_argument("--num-layers", type=int, default=1)
        parser.add_argument("--sampling-rate", type=float, default=0.01)
        parser.add_argument("--score-func", type=str, default="conve")
        parser.add_argument("--lbl_smooth", type=float, default=0.1)
        # fmt: on
    
    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            num_entities=args.num_entities,
            num_rels=args.num_rels,
            hidden_size=args.hidden_size,
            num_bases=args.num_bases,
            sampling_rate=args.sampling_rate,
            score_func=args.score_func,
            penalty=args.penalty,
            layers=args.num_layers,
            dropout=args.dropout,
            lbl_smooth=args.lbl_smooth,
        )

    def __init__(self, num_entities, num_rels, hidden_size, num_bases=0, layers=1,
                 sampling_rate=0.01, score_func="conve", penalty=0.001, dropout=0.0, lbl_smooth=0.1):
        BaseModel.__init__(self)
        GNNLinkPredict.__init__(self, score_func, hidden_size)
        activation = F.tanh
        self.model = CompGCN(num_entities, num_rels, num_bases, hidden_size//2, hidden_size, hidden_size, layers, dropout, activation)
        # self.emb = nn.Parameter(torch.Tensor(num_entities, hidden_size))
        # nn.init.xavier_uniform_(self.emb)
        self.emb = nn.Embedding(num_entities, hidden_size//2)
        self.sampling_rate = sampling_rate
        self.penalty = penalty
        self.num_rels = num_rels
        self.num_entities = num_entities
        self.cache_index = None
        self.lbl_smooth = lbl_smooth

    def add_reverse_edges(self, edge_index, edge_types):
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
        edge_types_rev = edge_types + self.num_rels
        edge_types = torch.cat([edge_types, edge_types_rev])
        return edge_index, edge_types

    def forward(self, edge_index, edge_types):
        # edge_index, edge_types = self.add_reverse_edges(edge_index, edge_types)
        reindexed_node, reindexed_edge_index = torch.unique(edge_index, return_inverse=True, sorted=True)
        self.cache_index = reindexed_node
        node_embed = self.emb(reindexed_node)
        node_embed, rel_embed = self.model(node_embed, reindexed_edge_index, edge_types)
        return node_embed, rel_embed

    def loss(self, data, split="train"):
        if split == "train":
            mask = data.train_mask
        elif split == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask
        edge_index, edge_types = data.edge_index[:, mask], data.edge_attr[mask]
        
        self.get_edge_set(edge_index, edge_types)
        batch_edges, batch_attr, samples, rels, labels = sampling_edge_uniform(edge_index, edge_types, self.edge_set, self.sampling_rate, self.num_rels, label_smoothing=self.lbl_smooth, num_entities=self.num_entities)
        node_embed, rel_embed = self.forward(batch_edges, batch_attr)

        sampled_nodes, reindexed_edges = torch.unique(samples, sorted=True, return_inverse=True)
        assert (self.cache_index == sampled_nodes).any()
        loss_n = self._loss(node_embed[reindexed_edges[0]], node_embed[reindexed_edges[1]], rel_embed[rels], labels) 
        loss_r = self.penalty * self._regularization([self.emb(sampled_nodes), rel_embed])
        return loss_n + loss_r
        
        # Full graph as input
        # sampled_nodes = torch.unique(samples)
        # node_embed, rel_embed = self.forward(edge_index, edge_types)
        # loss_n = self._loss(node_embed[samples[0]], node_embed[samples[1]], rel_embed[rels], labels)
        # loss_r = self.penalty * self._regularization([self.emb(sampled_nodes), rel_embed])
        # return loss_n + loss_r


    def predict(self, edge_index, edge_types):
        indices = torch.arange(0, self.num_entities).to(edge_index.device)
        x = self.emb(indices)
        # edge_index, edge_types = self.add_reverse_edges(edge_index, edge_types)
        node_embed, rel_embed = self.model(x, edge_index, edge_types)
        mrr, hits = cal_mrr(node_embed, rel_embed, edge_index, edge_types, scoring=self.scoring, protocol="raw", batch_size=500, hits=[1, 3, 10])
        return mrr, hits
