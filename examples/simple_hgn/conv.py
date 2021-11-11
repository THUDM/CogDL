import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import edge_softmax, get_activation
from cogdl.utils.spmm_utils import MultiHeadSpMM


class myGATConv(nn.Module):
    """
    cogdl implementation of Simple-HGN layer
    """

    def __init__(
        self,
        edge_feats,
        num_etypes,
        in_features,
        out_features,
        nhead,
        feat_drop=0.0,
        attn_drop=0.5,
        negative_slope=0.2,
        residual=False,
        activation=None,
        alpha=0.0,
    ):
        super(myGATConv, self).__init__()
        self.edge_feats = edge_feats
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.edge_emb = nn.Parameter(torch.zeros(size=(num_etypes, edge_feats)))  # nn.Embedding(num_etypes, edge_feats)

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * nhead))
        self.W_e = nn.Parameter(torch.FloatTensor(edge_feats, edge_feats * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))
        self.a_e = nn.Parameter(torch.zeros(size=(1, nhead, edge_feats)))

        self.mhspmm = MultiHeadSpMM()

        self.feat_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.act = None if activation is None else get_activation(activation)

        if residual:
            self.residual = nn.Linear(in_features, out_features * nhead)
        else:
            self.register_buffer("residual", None)
        self.reset_parameters()
        self.alpha = alpha

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.a_e)
        reset(self.W)
        reset(self.W_e)
        reset(self.edge_emb)

    def forward(self, graph, x, res_attn=None):
        x = self.feat_drop(x)
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0
        e = torch.matmul(self.edge_emb, self.W_e).view(-1, self.nhead, self.edge_feats)

        row, col = graph.edge_index
        tp = graph.edge_type
        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)[row]
        h_r = (self.a_r * h).sum(dim=-1)[col]
        h_e = (self.a_e * e).sum(dim=-1)[tp]
        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        # edge_attention: E * H
        edge_attention = edge_softmax(graph, edge_attention)
        edge_attention = self.dropout(edge_attention)
        if res_attn is not None:
            edge_attention = edge_attention * (1 - self.alpha) + res_attn * self.alpha

        # if check_mh_spmm() and next(self.parameters()).device.type != "cpu":
        #     if self.nhead > 1:
        #         h_prime = mh_spmm(graph, edge_attention, h)
        #         out = h_prime.view(h_prime.shape[0], -1)
        #     else:
        #         edge_weight = edge_attention.view(-1)
        #         with graph.local_graph():
        #             graph.edge_weight = edge_weight
        #             out = spmm(graph, h.squeeze(1))
        # else:
        #     with graph.local_graph():
        #         h_prime = []
        #         h = h.permute(1, 0, 2).contiguous()
        #         for i in range(self.nhead):
        #             edge_weight = edge_attention[:, i]
        #             graph.edge_weight = edge_weight
        #             hidden = h[i]
        #             assert not torch.isnan(hidden).any()
        #             h_prime.append(spmm(graph, hidden))
        #     out = torch.cat(h_prime, dim=1)
        out = self.mhspmm(graph, edge_attention, h)

        if self.residual:
            res = self.residual(x)
            out += res
        if self.act is not None:
            out = self.act(out)
        return out, edge_attention.detach()

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"
