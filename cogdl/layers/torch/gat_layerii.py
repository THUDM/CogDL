import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.utils import spmm_scatter, get_activation

from cogdl.utils import (
    EdgeSoftmax,
    MultiHeadSpMM,
    get_norm_layer,
    check_fused_gat,
    fused_gat_op,
)


class GATLayerST(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self, in_feats, out_feats, nhead=1, alpha=0.2, attn_drop=0.5, activation=None, residual=False, norm=None
    ):
        super(GATLayerST, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        self.alpha = alpha
        self.nhead = nhead

        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_feats)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_feats)))

        self.edge_softmax = EdgeSoftmax()
        self.mhspmm = MultiHeadSpMM()

        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.act = None if activation is None else get_activation(activation)
        self.norm = None if norm is None else get_norm_layer(norm, out_feats * nhead)

        if residual:
            self.residual = nn.Linear(in_feats, out_feats * nhead)
        else:
            self.register_buffer("residual", None)
        self.reset_parameters()

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.W)

    def forward(self, x, edge_index, edge_weight):
        support = self.linear(x)
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0

        row, col = edge_index

        out = spmm_scatter(row, col, edge_weight, support)

        # TODO: Don't support the Multi-Head ATT/ATT now, the gcn_layer is used in here.
  
        # Self-attention on the nodes - Shared attention mechanism
        # h_l = (self.a_l * h).sum(dim=-1)
        # h_r = (self.a_r * h).sum(dim=-1)
        #
        # if self.dropout.p == 0.0 and check_fused_gat():
        #     out = fused_gat_op(h_l, h_r, graph, self.alpha, h)
        #     out = out.view(out.shape[0], -1)
        # else:
        #     # edge_attention: E * H
        #     edge_attention = self.leakyrelu(h_l[row] + h_r[col])
        #     edge_attention = self.edge_softmax(graph, edge_attention)
        #     edge_attention = self.dropout(edge_attention)
        #
        #     out = self.mhspmm(graph, edge_attention, h)

        if self.residual:
            res = self.residual(x)
            out += res
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"
