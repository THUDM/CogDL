import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import row_normalization, spmm


class RGCNLayer(nn.Module):
    """
    Implementation of Relational-GCN in paper `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_

    Parameters
    ----------
    in_feats : int
        Size of each input embedding.
    out_feats : int
        Size of each output embedding.
    num_edge_type : int
        The number of edge type in knowledge graph.
    regularizer : str, optional
        Regularizer used to avoid overfitting, ``basis`` or ``bdd``, default : ``basis``.
    num_bases : int, optional
        The number of basis, only used when `regularizer` is `basis`, default : ``None``.
    self_loop : bool, optional
        Add self loop embedding if True, default : ``True``.
    dropout : float
    self_dropout : float, optional
        Dropout rate of self loop embedding, default : ``0.0``
    layer_norm : bool, optional
        Use layer normalization if True, default : ``True``
    bias : bool
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_edge_types,
        regularizer="basis",
        num_bases=None,
        self_loop=True,
        dropout=0.0,
        self_dropout=0.0,
        layer_norm=True,
        bias=True,
    ):
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
                torch.Tensor(num_edge_types, self.num_bases, self.block_in_feats * self.block_out_feats)
            )
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

    def forward(self, graph, x):
        if self.regularizer == "basis":
            h_list = self.basis_forward(graph, x)
        else:
            h_list = self.bdd_forward(graph, x)

        h_result = sum(h_list)
        h_result = F.dropout(h_result, p=self.dropout, training=self.training)
        if self.layer_norm is not None:
            h_result = self.layer_norm(h_result)
        if self.bias is not None:
            h_result = h_result + self.bias
        if self.self_loop is not None:
            h_result += F.dropout(torch.matmul(x, self.weight_self_loop), p=self.self_dropout, training=self.training)
        return h_result

    def basis_forward(self, graph, x):
        edge_type = graph.edge_attr

        if self.num_bases < self.num_edge_types:
            weight = torch.matmul(self.alpha, self.weight.view(self.num_bases, -1))
            weight = weight.view(self.num_edge_types, self.in_feats, self.out_feats)
        else:
            weight = self.weight

        edge_index = torch.stack(graph.edge_index)
        edge_weight = graph.edge_weight

        graph.row_norm()
        h = torch.matmul(x, weight)  # (N, d1) by (r, d1, d2) -> (r, N, d2)

        h_list = []
        for edge_t in range(self.num_edge_types):
            g = graph.__class__()
            edge_mask = edge_type == edge_t

            if edge_mask.sum() == 0:
                h_list.append(0)
                continue

            g.edge_index = edge_index[:, edge_mask]

            g.edge_weight = edge_weight[edge_mask]
            g.padding_self_loops()

            temp = spmm(graph, h[edge_t])
            h_list.append(temp)
            return h_list

    def bdd_forward(self, graph, x):
        edge_type = graph.edge_attr
        edge_index = torch.stack(graph.edge_index)
        _x = x.view(-1, self.num_bases, self.block_in_feats)

        edge_weight = torch.ones(edge_type.shape).to(x.device)
        edge_weight = row_normalization(x.shape[0], edge_index, edge_weight)

        h_list = []
        for edge_t in range(self.num_edge_types):
            _weight = self.weight[edge_t].view(self.num_bases, self.block_in_feats, self.block_out_feats)
            edge_mask = edge_type == edge_t
            _edge_index_t = edge_index.t()[edge_mask].t()
            h_t = torch.einsum("abc,bcd->abd", _x, _weight).reshape(-1, self.out_feats)
            h_t = spmm(_edge_index_t, edge_weight[edge_mask], h_t)
            h_list.append(h_t)

        return h_list
