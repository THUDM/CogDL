import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import check_mh_spmm, mh_spmm, mul_edge_softmax, spmm, get_activation, get_norm_layer


class GATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self, in_features, out_features, nhead=1, alpha=0.2, attn_drop=0.5, activation=None, residual=False, norm=None
    ):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.nhead = nhead

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))

        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.act = None if activation is None else get_activation(activation)
        self.norm = None if norm is None else get_norm_layer(norm, out_features * nhead)

        if residual:
            self.residual = nn.Linear(in_features, out_features * nhead)
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

    def forward(self, graph, x):
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0

        row, col = graph.edge_index
        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)[row]
        h_r = (self.a_r * h).sum(dim=-1)[col]
        edge_attention = self.leakyrelu(h_l + h_r)
        # edge_attention: E * H
        edge_attention = mul_edge_softmax(graph, edge_attention)
        edge_attention = self.dropout(edge_attention)

        if check_mh_spmm() and next(self.parameters()).device.type != "cpu":
            if self.nhead > 1:
                h_prime = mh_spmm(graph, edge_attention, h)
                out = h_prime.view(h_prime.shape[0], -1)
            else:
                edge_weight = edge_attention.view(-1)
                with graph.local_graph():
                    graph.edge_weight = edge_weight
                    out = spmm(graph, h.squeeze(1))
        else:
            with graph.local_graph():
                h_prime = []
                h = h.permute(1, 0, 2).contiguous()
                for i in range(self.nhead):
                    edge_weight = edge_attention[:, i]
                    graph.edge_weight = edge_weight
                    hidden = h[i]
                    assert not torch.isnan(hidden).any()
                    h_prime.append(spmm(graph, hidden))
            out = torch.cat(h_prime, dim=1)

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
