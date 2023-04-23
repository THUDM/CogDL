import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import (
    EdgeSoftmax,
    MultiHeadSpMM,
    get_activation,
    get_norm_layer,
)


class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = activation if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None
        self.feat_drop = nn.Dropout(feat_drop)
        
        if num_layers == 1:
            self.gat_layers.append(
                GATLayer(
                    in_dim, out_dim, nhead=nhead_out, 
                    attn_drop=attn_drop, 
                    alpha=negative_slope, 
                    residual=last_residual, 
                    norm=last_norm, 
                    activation=last_activation,
                )
            )
        else:
            # input projection (no residual)
            self.gat_layers.append(
                GATLayer(
                    in_dim, num_hidden, nhead,
                    attn_drop=attn_drop, 
                    alpha=negative_slope, 
                    residual=residual, 
                    activation=activation, 
                    norm=norm
                )
            )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(
                    GATLayer(
                        num_hidden * nhead, num_hidden, nhead=nhead,
                        attn_drop=attn_drop, 
                        alpha=negative_slope, 
                        residual=residual, 
                        activation=activation, 
                        norm=norm
                    )
                )
            # output projection
            self.gat_layers.append(
                GATLayer(
                    num_hidden * nhead, out_dim, 
                    nhead=nhead_out,
                    attn_drop=attn_drop, 
                    alpha=negative_slope, 
                    residual=last_residual, 
                    activation=last_activation, 
                    norm=last_norm
                )
            )
        self.head = nn.Identity()
    
    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = self.feat_drop(h)
            h = self.gat_layers[l](g, h)
            hidden_list.append(h)
            # h = h.flatten(1)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


class GATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self, in_feats, out_feats, nhead=1, alpha=0.2, attn_drop=0.5, activation=None, residual=False, norm=None
    ):
        super(GATLayer, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        self.alpha = alpha
        self.nhead = nhead

        self.W = nn.Parameter(torch.FloatTensor(in_feats, out_feats * nhead))
        self.bias = nn.Parameter(torch.FloatTensor(out_feats * nhead,))

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
        # def reset(tensor):
        #     stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        #     tensor.data.uniform_(-stdv, stdv)

        # reset(self.a_l)
        # reset(self.a_r)
        # reset(self.W)

        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_normal_(self.residual.weight, gain=gain)

    def forward(self, graph, x):
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0

        row, col = graph.edge_index
        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)
        h_r = (self.a_r * h).sum(dim=-1)

        # edge_attention: E * H
        edge_attention = self.leakyrelu(h_l[row] + h_r[col])
        edge_attention = self.edge_softmax(graph, edge_attention)
        edge_attention = self.dropout(edge_attention)

        out = self.mhspmm(graph, edge_attention, h)

        if self.bias is not None:
            out = out + self.bias

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
