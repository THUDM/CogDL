import random
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from cogdl import function as BF
from cogdl.backend import BACKEND

if BACKEND == "jittor":
    from cogdl.operators.jittor.sample import coo2csr_cpu, coo2csr_cpu_index
elif BACKEND == "torch":
    from cogdl.operators.torch.sample import coo2csr_cpu, coo2csr_cpu_index
else:
    raise ("Unsupported backend:", BACKEND)


def get_degrees(row, col, num_nodes=None):
    device = BF.device(row)
    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1
    b = BF.ones(col.shape[0], device=device)
    out = BF.zeros(num_nodes, device=device)
    degrees = BF.scatter_add_(out, dim=0, index=row, src=b)
    return degrees.float()


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    row, col = edge_index
    device = BF.device(row)
    if edge_weight is None:
        edge_weight = BF.ones(edge_index[0].shape[0], device=device)
    if num_nodes is None:
        num_nodes = BF.max(edge_index) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    self_weight = BF.full((num_nodes,), fill_value, dtype=edge_weight.dtype, device=BF.device(edge_weight))
    loop_index = BF.arange(0, N, dtype=row.dtype, device=device)
    row = BF.cat([row, loop_index])
    col = BF.cat([col, loop_index])
    edge_index = BF.stack([row, col])
    edge_weight = BF.cat([edge_weight, self_weight])
    return edge_index, edge_weight


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    device = BF.device(edge_index[0])
    row, col = edge_index[0], edge_index[1]

    if edge_weight is None:
        edge_weight = BF.ones(row.shape[0], device=device)
    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    mask = row != col

    loop_index = BF.arange(0, N, dtype=row.dtype, device=BF.device(row))
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    _row = BF.cat([row[mask], loop_index[0]])
    _col = BF.cat([col[mask], loop_index[1]])
    # edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    # inv_mask = ~mask
    inv_mask = BF.logical_not(mask)

    loop_weight = BF.full((N,), fill_value, dtype=edge_weight.dtype, device=BF.device(edge_weight))
    remaining_edge_weight = edge_weight[inv_mask]
    if remaining_edge_weight.numel() > 0:
        loop_weight[row[inv_mask]] = remaining_edge_weight
    edge_weight = BF.cat([edge_weight[mask], loop_weight], dim=0)

    return (_row, _col), edge_weight


def row_normalization(num_nodes, row, col, val=None):
    device = BF.device(row)
    if val is None:
        val = BF.ones(row.shape[0], device=device)
    row_sum = get_degrees(row, col, num_nodes)
    row_sum_inv = row_sum.pow(-1).view(-1)
    row_sum_inv[BF.isinf(row_sum_inv)] = 0
    return val * row_sum_inv[row]


def symmetric_normalization(num_nodes, row, col, val=None):
    device = BF.device(row)
    if val is None:
        val = BF.ones(row.shape[0], device=device)
    row_sum = get_degrees(row, col, num_nodes)
    row_sum_inv_sqrt = row_sum.pow(-0.5)
    row_sum_inv_sqrt[row_sum_inv_sqrt == float("inf")] = 0
    return row_sum_inv_sqrt[col] * val * row_sum_inv_sqrt[row]


def _coo2csr(edge_index, data, num_nodes=None, ordered=False, return_index=False):
    if ordered:
        return sorted_coo2csr(edge_index[0], edge_index[1], data, return_index=return_index)
    if num_nodes is None:
        num_nodes = BF.max(edge_index) + 1
    device = BF.device(edge_index[0])
    sorted_index = BF.argsort(edge_index[0])
    sorted_index = sorted_index.long()
    edge_index = edge_index[:, sorted_index]
    indices = edge_index[1]

    row = edge_index[0]
    indptr = BF.zeros(num_nodes + 1, dtype=BF.dtype_dict("int32"), device=device)
    elements, counts = BF.unique(row, return_counts=True)
    elements = elements.long() + 1
    indptr[elements] = BF.type_as(counts, indptr)
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
        num_nodes = BF.max(BF.stack(row, col)).item() + 1
    if coo2csr_cpu is None:
        return _coo2csr(BF.stack([row, col]), data, num_nodes)
    device = BF.device(row)
    row = BF.cpu(row.long())
    col = BF.cpu(col.long())
    data = BF.cpu(data.float())
    indptr, indices, data = coo2csr_cpu(row, col, data, num_nodes)
    return BF.to(indptr, device), BF.to(indices, device), BF.to(data, device)


def coo2csr_index(row, col, num_nodes=None):
    if num_nodes is None:
        num_nodes = BF.max(BF.stack([row, col])).item() + 1
    if coo2csr_cpu_index is None:
        return _coo2csr(BF.stack([row, col]), None, num_nodes=num_nodes, return_index=True)
    device = BF.device(row)
    row = BF.cpu(row.long())
    col = BF.cpu(col.long())
    indptr, reindex = coo2csr_cpu_index(row, col, num_nodes)
    return BF.to(indptr, device), BF.to(reindex, device)


def sorted_coo2csr(row, col, data, num_nodes=None, return_index=False):
    indptr = BF.bincount(row)
    indptr = indptr.cumsum(dim=0)
    zero = BF.zeros(1, device=BF.device(indptr))
    indptr = BF.cat([zero, indptr])
    if return_index:
        return indptr, BF.arange(0, row.shape[0])
    return indptr, col, data


def coo2csc(row, col, data, num_nodes=None, sorted=False):
    return coo2csr(col, row, data, num_nodes, sorted)


def csr2csc(indptr, indices, data=None):
    device = BF.device(indices)
    indptr = BF.cpu(indptr).numpy()
    indices = BF.cpu(indices).numpy()
    num_nodes = indptr.shape[0] - 1
    if data is None:
        data = np.ones(indices.shape[0])
    else:
        data = BF.cpu(data).numpy()
    adj = sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    adj = adj.tocsc()
    data = BF.as_tensor(adj.data, device=device)
    col_indptr = BF.as_tensor(adj.indptr, device=device)
    row_indices = BF.as_tensor(adj.indices, device=device)
    return col_indptr, row_indices, data


def csr2coo(indptr, indices, data):
    num_nodes = indptr.size(0) - 1
    row = BF.arange(num_nodes, device=BF.device(indptr))
    row_count = indptr[1:] - indptr[:-1]
    row = BF.repeat_interleave(row, row_count)
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
    device = BF.device(row)
    if BF.is_tensor(row):
        row = BF.cpu(row).numpy()
    if BF.is_tensor(col):
        col = BF.cpu(col).numpy()
    indices = np.lexsort((col, row))
    row = BF.to(BF.from_numpy(row[indices]).long(), device)
    col = BF.to(BF.from_numpy(col[indices]).long(), device)

    num = col.shape[0] + 1
    idx = BF.full((num,), -1, dtype=BF.dtype_dict("long"), device=device)
    max_num = max(row.max(), col.max()) + 100
    idx[1:] = (row + 1) * max_num + col
    mask = idx[1:] > idx[:-1]

    if mask.all():
        return row, col, value
    row = row[mask]
    if value is not None:
        _value = BF.zeros(row.shape[0], dtype=BF.dtype_dict("float"), device=device)
        value = BF.scatter_add_(_value, dim=0, src=value, index=col)
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
    row, col = BF.cat([row, col], dim=0), BF.cat([col, row], dim=0)
    row, col, _ = coalesce(row, col, None)
    edge_index = BF.stack([row, col])
    return edge_index


def negative_edge_sampling(
    edge_index: Union[Tuple, BF.dtype_dict("tensor")],  # noqa
    num_nodes: Optional[int] = None,
    num_neg_samples: Optional[int] = None,
    undirected: bool = False,
):
    if num_nodes is None:
        num_nodes = len(BF.unique(edge_index))
    if num_neg_samples is None:
        num_neg_samples = edge_index[0].shape[0]

    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index[1].shape[0])

    row, col = edge_index
    unique_pair = row * num_nodes + col

    num_samples = int(num_neg_samples * abs(1 / (1 - 1.1 * row.size(0) / size)))
    sample_result = BF.LongTensor(random.sample(range(size), min(num_samples, num_samples)))
    mask = BF.from_numpy(np.isin(sample_result, BF.to(unique_pair, "cpu"))).bool()
    selected = BF.to(sample_result[~mask][:num_neg_samples], row)

    row = selected // num_nodes
    col = selected % num_nodes
    return BF.stack([row, col]).long()
