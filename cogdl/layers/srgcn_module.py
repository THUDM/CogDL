from functools import reduce
from scipy.special import iv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse.tensor import SparseTensor
from torch_sparse import spspmm, spmm, matmul
from torch_geometric.utils import degree


# ==========
# Attention
# ==========


class NodeAttention(nn.Module):
    def __init__(self, in_feat):
        super(NodeAttention, self).__init__()
        self.p = nn.Parameter(torch.zeros(in_feat, 1))
        self.b = nn.Parameter(torch.zeros(1, ))
        nn.init.xavier_normal_(self.p.data, gain=1.414)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, edge_index, edge_attr):
        device = x.device
        N, dim = x.shape
        diag_val = torch.mm(x, self.p) + self.b

        # add sigmoid
        diag_val = F.sigmoid(diag_val)
        # add dropout
        self.dropout(diag_val)

        diag_ind = torch.LongTensor([range(N)] * 2).to(device)
        _, adj_mat_val = spspmm(edge_index, edge_attr, diag_ind, diag_val.view(-1), N, N, N, True)

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

        # add dropout
        p_val = self.dropout(p_val)
        q_val = self.dropout(q_val)
        # --------------------

        diag_ind = torch.LongTensor([range(N)] * 2).to(device)
        _, p_adj_mat_val = spspmm(edge_index, edge_attr_t, diag_ind, p_val.view(-1), N, N, N, True)
        _, q_adj_mat_val = spspmm(diag_ind, q_val.view(-1), edge_index, edge_attr, N, N, N, True)
        return edge_index, p_adj_mat_val + q_adj_mat_val


class Identity(nn.Module):
    def __init__(self, in_feat):
        super(Identity, self).__init__()

    def forward(self, x, edge_index, edge_attr):
        return edge_index, edge_attr


class Gaussian(nn.Module):
    def __init__(self, in_feat):
        super(Gaussian, self).__init__()
        self.mu = 0.2
        self.theta = 1.
        self.steps = 4

    def forward(self, x, edge_index, edge_attr):
        N = x.shape[0]
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        adj = torch.sparse_coo_tensor(edge_index, deg_inv[row] * edge_attr, size=(N, N))
        identity = torch.sparse_coo_tensor([range(N)] * 2, torch.ones(N), size=(N, N)).to(x.device)
        laplacian = identity - adj

        t0 = identity
        t1 = laplacian - self.mu * identity
        t1 = t1.mm(t1.to_dense()).to_sparse()
        l_x = -0.5 * (t1 - identity)
        # l_x = -0.5 * ((laplacian - self.mu * identity).pow(2) - identity)

        ivs = [iv(i, self.theta) for i in range(self.steps)]
        ivs[1:] = [(-1) ** i * 2 * x for i, x in enumerate(ivs[1:])]
        ivs = torch.tensor(ivs).to(x.device)
        result = [t0, l_x]
        for i in range(2, self.steps):
            result.append(2 * l_x.mm(result[i - 1].to_dense()).to_sparse().sub(result[i - 2]))

            # adj_ind, adj_val = spspmm(l_x._indices(), l_x._values(), result[i-1]._indices(), result[i-1]._values(), N, N, N, True)
            # obj = torch.sparse_coo_tensor(adj_ind, adj_val * 2, size=(N, N))
            # obj = obj.sub(result[i-2])
            # result.append(obj)

        result = [result[i] * ivs[i] for i in range(self.steps)]

        def fn(x, y):
            return x.add(y)

        res = reduce(fn, result)
        return res._indices(), res._values()


class PPR(nn.Module):
    def __init__(self, in_feat):
        super(PPR, self).__init__()
        self.alpha = 0.4
        self.steps = 4
        self.epsilon = 0.005

    def forward(self, x, edge_index, edge_attr):
        device = x.device
        edge_attr_t = edge_attr
        N = x.size(0)

        thetas = [self.alpha * (1 - self.alpha) ** i for i in range(0, self.steps)]
        res_edges = [edge_index, edge_index]
        res_val = [torch.ones(edge_attr_t.shape[0]).to(device), edge_attr_t]

        for i in range(1, self.steps - 1):
            adj_ind, adj_val = spspmm(edge_index, edge_attr_t, res_edges[-1], res_val[-1], N, N, N, True)

            selected = adj_val >= self.epsilon
            adj_val = adj_val[selected]
            adj_ind = torch.stack([adj_ind[0][selected], adj_ind[1][selected]], dim=0)

            res_edges.append(adj_ind)
            res_val.append(adj_val)
        result = [torch.sparse_coo_tensor(indices=res_edges[i], values=res_val[i], size=(N, N)) * thetas[i] for i in range(self.steps)]

        # --------------- use matmul ---------
        # identity = self.alpha * torch.sparse_coo_tensor([range(N)] * 2, torch.ones(N), size=(N, N)).to(x.device)
        # adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(N, N), value=edge_attr_t)
        #
        # result = [identity, adj]
        # thetas = [self.alpha * (1 - self.alpha)**i for i in range(0, self.steps)]
        # for _ in range(2, self.steps):
        #     adj_t = matmul(adj, result[-1])
        #     result.append(adj_t)
        # assert len(result) == self.steps
        # assert len(thetas) == self.steps

        # result[1:] = [result[i].to_torch_sparse_coo_tensor() * thetas[i] for i in range(1, self.steps)]
        # ---------------------------

        def fn(a, b):
            return a.add(b)

        res = reduce(fn, result)
        return res._indices(), res._values()


class HeatKernel(nn.Module):
    def __init__(self, in_feat):
        super(HeatKernel, self).__init__()
        self.t = nn.Parameter(torch.zeros(1, ))

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        edge_attr_t = self.t * edge_attr * deg_inv[col] - self.t
        return edge_index, edge_attr_t.exp()


class LayerSample(nn.Module):
    def __init__(self, in_feat):
        super(LayerSample, self).__init__()
        self.sample_rate = torch.nn.Parameter(torch.zeros(1, ))

    def forward(self, x, edge_index, edge_attr):
        device = x.device
        sample_rate = F.sigmoid(self.sample_rate)
        _sample_rate = sample_rate.detach().numpy()
        N = x.shape[0]
        sampled = torch.rand(N) > _sample_rate[0]

        edge_attr = torch.tensor([edge_attr[i]
                                  for i in range(edge_index.shape[0])
                                  if sampled[edge_index[0][i]] and sampled[edge_index[1][i]]]).to(device)
        edge_attr = edge_attr / sample_rate

        edge_index = torch.tensor([[edge_index[0][i], edge_index[1][i]]
                                   for i in range(edge_index.shape[0])
                                   if sampled[edge_index[0][i]] and sampled[edge_index[1][i]]]).to(device)
        return edge_index, edge_attr


def act_attention(attn_type):
    if attn_type == "identity":
        return Identity
    elif attn_type == "node":
        return NodeAttention
    elif attn_type == "edge":
        return EdgeAttention
    elif attn_type == "gaussian":
        return Gaussian
    elif attn_type == "ppr":
        return PPR
    elif attn_type == "heat":
        return HeatKernel
    elif attn_type == "layer_sample":
        return LayerSample
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
        # col = rownorm[edge_index[1]]
        edge_attr_t = row * edge_attr

        return edge_attr_t


class RowSoftmax(nn.Module):
    def __init__(self):
        super(RowSoftmax, self).__init__()

    def forward(self, edge_index, edge_attr, N):
        device = edge_attr.device
        edge_attr = F.leaky_relu(edge_attr)
        edge_attr_t = torch.exp(edge_attr)
        ones = torch.ones(N, 1, device=device)
        rownorm = 1. / spmm(edge_index, edge_attr_t, N, N, ones).view(-1)
        row = rownorm[edge_index[0]]
        # col = rownorm[edge_index[1]]
        edge_attr_t = row * edge_attr_t

        return edge_attr_t


class ColumnUniform(nn.Module):
    def __init__(self):
        super(ColumnUniform, self).__init__()

    def forward(self, edge_index, edge_attr, N):
        device = edge_attr.device
        ones = torch.ones(N, 1, device=device)
        rownorm = 1. / spmm(edge_index, edge_attr, N, N, ones).view(-1)
        # row = rownorm[edge_index[0]]
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


class ClusterGCNNorm(nn.Module):
    def __init__(self):
        super(ClusterGCNNorm, self).__init__()
        self.lmbda = 0.2

    def forward(self, edge_index, edge_attr, N):
        row, col = edge_index
        deg = degree(col, N, dtype=torch.float)
        deg_inv = deg.pow(-1)
        adj_attr = deg_inv[row] * edge_attr
        diag = row == col
        diag_attr = adj_attr[diag]
        diag_attr = diag_attr + self.lmbda * diag_attr
        adj_attr[diag] = diag_attr
        return adj_attr


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
    elif norm_type == "clusternorm":
        return ClusterGCNNorm
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
