import random
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from cogdl.operators.sample import coo2csr_cpu, coo2csr_cpu_index


def get_degrees(row, col, num_nodes=None):
    device = row.device
    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1
    b = torch.ones(col.shape[0], device=device)
    out = torch.zeros(num_nodes, device=device)
    degrees = out.scatter_add_(dim=0, index=row, src=b)
    return degrees.float()


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    row, col = edge_index
    device = row.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index[0].shape[0]).to(device)
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    self_weight = torch.full((num_nodes,), fill_value, dtype=edge_weight.dtype).to(edge_weight.device)
    loop_index = torch.arange(0, N, dtype=row.dtype, device=device)
    row = torch.cat([row, loop_index])
    col = torch.cat([col, loop_index])
    edge_index = torch.stack([row, col])
    edge_weight = torch.cat([edge_weight, self_weight])
    return edge_index, edge_weight


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    device = edge_index[0].device
    row, col = edge_index[0], edge_index[1]

    if edge_weight is None:
        edge_weight = torch.ones(row.shape[0], device=device)
    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    mask = row != col

    loop_index = torch.arange(0, N, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    _row = torch.cat([row[mask], loop_index[0]])
    _col = torch.cat([col[mask], loop_index[1]])
    # edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    inv_mask = ~mask

    loop_weight = torch.full((N,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device)
    remaining_edge_weight = edge_weight[inv_mask]
    if remaining_edge_weight.numel() > 0:
        loop_weight[row[inv_mask]] = remaining_edge_weight
    edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    return (_row, _col), edge_weight


def row_normalization(num_nodes, row, col, val=None):
    device = row.device
    if val is None:
        val = torch.ones(row.shape[0], device=device)
    row_sum = get_degrees(row, col, num_nodes)
    row_sum_inv = row_sum.pow(-1).view(-1)
    row_sum_inv[torch.isinf(row_sum_inv)] = 0
    return val * row_sum_inv[row]


def symmetric_normalization(num_nodes, row, col, val=None):
    device = row.device
    if val is None:
        val = torch.ones(row.shape[0]).to(device)
    row_sum = get_degrees(row, col, num_nodes)
    row_sum_inv_sqrt = row_sum.pow(-0.5)
    row_sum_inv_sqrt[row_sum_inv_sqrt == float("inf")] = 0
    return row_sum_inv_sqrt[col] * val * row_sum_inv_sqrt[row]


def _coo2csr(edge_index, data, num_nodes=None, ordered=False, return_index=False):
    if ordered:
        return sorted_coo2csr(edge_index[0], edge_index[1], data, return_index=return_index)
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    device = edge_index[0].device
    sorted_index = torch.argsort(edge_index[0])
    sorted_index = sorted_index.long()
    edge_index = edge_index[:, sorted_index]
    indices = edge_index[1]

    row = edge_index[0]
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int32, device=device)
    elements, counts = torch.unique(row, return_counts=True)
    elements = elements.long() + 1
    indptr[elements] = counts.to(indptr.dtype)
    indptr = indptr.cumsum(dim=0)

    if return_index:
        return indptr, sorted_index
    if data is not None:
        data = data[sorted_index]
    return indptr, indices, data


def coo2csr(row, col, data, num_nodes=None, ordered=False):
    if ordered:
        indptr, indices, data = sorted_coo2csr(row, col, data)
        return indptr, indices, data
    if num_nodes is None:
        num_nodes = torch.max(torch.stack(row, col)).item() + 1
    if coo2csr_cpu is None:
        return _coo2csr(torch.stack([row, col]), data, num_nodes)
    device = row.device
    row = row.long().cpu()
    col = col.long().cpu()
    data = data.float().cpu()
    indptr, indices, data = coo2csr_cpu(row, col, data, num_nodes)
    return indptr.to(device), indices.to(device), data.to(device)


def coo2csr_index(row, col, num_nodes=None):
    if num_nodes is None:
        num_nodes = torch.max(torch.stack([row, col])).item() + 1
    if coo2csr_cpu_index is None:
        return _coo2csr(torch.stack([row, col]), None, num_nodes=num_nodes, return_index=True)
    device = row.device
    row = row.long().cpu()
    col = col.long().cpu()
    indptr, reindex = coo2csr_cpu_index(row, col, num_nodes)
    return indptr.to(device), reindex.to(device)


def sorted_coo2csr(row, col, data, num_nodes=None, return_index=False):
    indptr = torch.bincount(row)
    indptr = indptr.cumsum(dim=0)
    zero = torch.zeros(1, device=indptr.device)
    indptr = torch.cat([zero, indptr])
    if return_index:
        return indptr, torch.arange(0, row.shape[0])
    return indptr, col, data


def coo2csc(row, col, data, num_nodes=None, sorted=False):
    return coo2csr(col, row, data, num_nodes, sorted)


def csr2csc(indptr, indices, data=None):
    device = indices.device
    indptr = indptr.cpu().numpy()
    indices = indices.cpu().numpy()
    num_nodes = indptr.shape[0] - 1
    if data is None:
        data = np.ones(indices.shape[0])
    else:
        data = data.cpu().numpy()
    adj = sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    adj = adj.tocsc()
    data = torch.as_tensor(adj.data, device=device)
    col_indptr = torch.as_tensor(adj.indptr, device=device)
    row_indices = torch.as_tensor(adj.indices, device=device)
    return col_indptr, row_indices, data


def csr2coo(indptr, indices, data):
    num_nodes = indptr.size(0) - 1
    row = torch.arange(num_nodes, device=indptr.device)
    row_count = indptr[1:] - indptr[:-1]
    row = row.repeat_interleave(row_count)
    return row, indices, data


def remove_self_loops(indices, values=None):
    row, col = indices
    mask = indices[0] != indices[1]
    row = row[mask]
    col = col[mask]
    if values is not None:
        values = values[mask]
    return (row, col), values


def coalesce(row, col, value=None):
    device = row.device
    if torch.is_tensor(row):
        row = row.cpu().numpy()
    if torch.is_tensor(col):
        col = col.cpu().numpy()
    indices = np.lexsort((col, row))
    row = torch.from_numpy(row[indices]).long().to(device)
    col = torch.from_numpy(col[indices]).long().to(device)

    num = col.shape[0] + 1
    idx = torch.full((num,), -1, dtype=torch.long).to(device)
    max_num = max(row.max(), col.max()) + 100
    idx[1:] = (row + 1) * max_num + col
    mask = idx[1:] > idx[:-1]

    if mask.all():
        return row, col, value
    row = row[mask]
    if value is not None:
        _value = torch.zeros(row.shape[0], dtype=torch.float).to(device)
        value = _value.scatter_add_(dim=0, src=value, index=col)
    col = col[mask]
    return row, col, value


def to_undirected(edge_index, num_nodes=None):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    row, col, _ = coalesce(row, col, None)
    edge_index = torch.stack([row, col])
    return edge_index


def negative_edge_sampling(
    edge_index: Union[Tuple, torch.Tensor],
    num_nodes: Optional[int] = None,
    num_neg_samples: Optional[int] = None,
    undirected: bool = False,
):
    if num_nodes is None:
        num_nodes = len(torch.unique(edge_index))
    if num_neg_samples is None:
        num_neg_samples = edge_index[0].shape[0]

    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index[1].shape[0])

    row, col = edge_index
    unique_pair = row * num_nodes + col

    num_samples = int(num_neg_samples * abs(1 / (1 - 1.1 * row.size(0) / size)))
    sample_result = torch.LongTensor(random.sample(range(size), min(num_samples, num_samples)))
    mask = torch.from_numpy(np.isin(sample_result, unique_pair.to("cpu"))).to(torch.bool)
    selected = sample_result[~mask][:num_neg_samples].to(row.device)

    row = selected // num_nodes
    col = selected % num_nodes
    return torch.stack([row, col]).long()
