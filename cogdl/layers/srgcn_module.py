from functools import reduce
from scipy.special import iv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm, spmm
from torch_geometric.utils import degree


# ==========
# Attention
# ==========


class NodeAttention(nn.Module):
    def __init__(self, in_feat):
        super(NodeAttention, self).__init__()
        self.p = nn.Linear(in_feat, 1)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x, edge_index, edge_attr):
        device = x.device
        N, dim = x.shape
        diag_val = self.p(x)
        diag_val = F.sigmoid(diag_val)
        self.dropout(diag_val)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        edge_attr_t = deg_inv[row] * edge_attr

        diag_ind = torch.LongTensor([range(N)] * 2).to(device)
        _, adj_mat_val = spspmm(edge_index, edge_attr_t, diag_ind, diag_val.view(-1), N, N, N, True)
        return edge_index, adj_mat_val


class EdgeAttention(nn.Module):
    def __init__(self, in_feat):
        super(EdgeAttention, self).__init__()
        self.p = nn.Linear(in_feat, 1)
        self.q = nn.Linear(in_feat, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_attr):
        device = x.device
        N, dim = x.shape

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        edge_attr_t = deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col]

        p_val = F.relu(self.p(x))
        q_val = F.relu(self.q(x))

        p_val = self.dropout(p_val)
        q_val = self.dropout(q_val)

        p_adj_mat_val = edge_attr_t * p_val.view(-1)[edge_index[1]]
        q_adj_mat_val = edge_attr_t * q_val.view(-1)[edge_index[0]]
        return edge_index, p_adj_mat_val + q_adj_mat_val


class Identity(nn.Module):
    def __init__(self, in_feat):
        super(Identity, self).__init__()

    def forward(self, x, edge_index, edge_attr):
        return edge_index, edge_attr


# class Gaussian(nn.Module):
#     def __init__(self, in_feat):
#         super(Gaussian, self).__init__()
#         self.mu = 0.2
#         self.theta = 1.
#         self.steps = 4

#     def forward(self, x, edge_index, edge_attr):
#         N = x.shape[0]
#         row, col = edge_index
#         deg = degree(row, x.size(0), dtype=x.dtype)
#         deg_inv = deg.pow(-1)
#         adj = torch.sparse_coo_tensor(edge_index, deg_inv[row] * edge_attr , size=(N, N))
#         identity = torch.sparse_coo_tensor([range(N)] * 2, torch.ones(N), size=(N, N)).to(x.device)
#         laplacian = identity - adj

#         t0 = identity
#         t1 = laplacian - self.mu * identity
#         t1 = t1.mm(t1.to_dense()).to_sparse()
#         l_x = -0.5 * (t1 - identity)
#         # l_x = -0.5 * ((laplacian - self.mu * identity).pow(2) - identity)

#         ivs = [iv(i, self.theta) for i in range(self.steps)]
#         ivs[1:] = [(-1) ** i * 2 * x for i, x in enumerate(ivs[1:])]
#         ivs = torch.tensor(ivs).to(x.device)
#         result = [t0, l_x]
#         for i in range(2, self.steps):
#             result.append(2*l_x.mm(result[i-1].to_dense()).to_sparse().sub(result[i-2]))

#         result = [result[i] * ivs[i] for i in range(self.steps)]

#         def fn(x, y):
#             return x.add(y)
#         res = reduce(fn, result)

#         return res._indices(), res._values()


class PPR(nn.Module):
    def __init__(self, in_feat):
        super(PPR, self).__init__()
        self.alpha = 0.4
        self.steps = 4

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        edge_attr_t = deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col]

        N = x.size(0)
        adj = torch.sparse_coo_tensor(edge_index, edge_attr_t, size=(N, N))

        theta = self.alpha * (1 - self.alpha)
        result = [theta * adj]

        for i in range(1, self.steps-1):
            theta = theta * (1 - self.alpha)
            adj_ind, adj_val = spspmm(edge_index, edge_attr_t, result[i-1]._indices(), result[i-1]._values(), N, N, N, True)
            result.append(torch.sparse_coo_tensor(adj_ind, adj_val, size=(N, N)))

        identity = torch.sparse_coo_tensor([range(N)] * 2, torch.ones(N), size=(N, N)).to(x.device)
        result.append(self.alpha * identity)

        def fn(x, y):
            return x.add(y)

        res = reduce(fn, result)
        return res._indices(), res._values()


class HeatKernel(nn.Module):
    def __init__(self, in_feat):
        super(HeatKernel, self).__init__()
        self.t = nn.Parameter(torch.zeros(1,))

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        edge_attr_t = self.t * edge_attr * deg_inv[col] - self.t
        return edge_index, edge_attr_t.exp()


def act_attention(attn_type):
    if attn_type == "identity":
        return Identity
    elif attn_type == "node":
        return NodeAttention
    elif attn_type == "edge":
        return EdgeAttention
    elif attn_type == "ppr":
        return PPR
    elif attn_type == "heat":
        return HeatKernel
    else:
        raise ValueError("no such attention type")


# ===============
# Normalization
# ===============


class NormIdentity(nn.Module):
    def __init__(self):
        super(NormIdentity, self).__init__()

    def forward(self, edge_index, edge_attr, N):
        return edge_attr


class RowUniform(nn.Module):
    def __init__(self):
        super(RowUniform, self).__init__()

    def forward(self, edge_index, edge_attr, N):
        device = edge_attr.device
        ones = torch.ones(N, 1, device=device)
        rownorm = 1. / spmm(edge_index, edge_attr, N, N, ones).view(-1)
        row = rownorm[edge_index[0]]
        edge_attr_t = row * edge_attr

        return edge_attr_t


class RowSoftmax(nn.Module):
    def __init__(self):
        super(RowSoftmax, self).__init__()

    def forward(self, edge_index, edge_attr, N):
        device = edge_attr.device
        edge_attr_t = torch.exp(edge_attr)
        ones = torch.ones(N, 1, device=device)
        rownorm = 1. / spmm(edge_index, edge_attr_t, N, N, ones).view(-1)
        row = rownorm[edge_index[0]]
        edge_attr_t = row * edge_attr_t

        return edge_attr_t


class ColumnUniform(nn.Module):
    def __init__(self):
        super(ColumnUniform, self).__init__()

    def forward(self, edge_index, edge_attr, N):
        device = edge_attr.device
        ones = torch.ones(N, 1, device=device)
        rownorm = 1. / spmm(edge_index, edge_attr, N, N, ones).view(-1)
        col = rownorm[edge_index[1]]
        edge_attr_t = col * edge_attr

        return edge_attr_t


class SymmetryNorm(nn.Module):
    def __init__(self):
        super(SymmetryNorm, self).__init__()

    def forward(self, edge_index, edge_attr, N):
        device = edge_attr.device
        ones = torch.ones(N, 1, device=device)
        rownorm = spmm(edge_index, edge_attr, N, N, ones).view(-1).pow(-0.5)
        row = rownorm[edge_index[0]]
        col = rownorm[edge_index[1]]
        edge_attr_t = row * edge_attr * col

        return edge_attr_t


def act_normalization(norm_type):
    if norm_type == "identity":
        return NormIdentity
    elif norm_type == "row_uniform":
        return RowUniform
    elif norm_type == "row_softmax":
        return RowSoftmax
    elif norm_type == "col_uniform":
        return ColumnUniform
    elif norm_type == "symmetry":
        return SymmetryNorm
    else:
        raise ValueError("no such normalization type")


# ============
# activation 
# ============

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")