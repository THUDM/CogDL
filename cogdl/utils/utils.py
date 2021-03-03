import errno
import itertools
import os
import os.path as osp
import random
import shutil
from collections import defaultdict
from typing import Optional
from urllib import request

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tabulate import tabulate


class ArgClass(object):
    def __init__(self):
        pass


def build_args_from_dict(dic):
    args = ArgClass()
    for key, value in dic.items():
        args.__setattr__(key, value)
    return args


def untar(path, fname, deleteTar=True):
    """
    Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print("unpacking " + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, name=None, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        name (string): saved filename.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    if log:
        print("Downloading", url)

    makedirs(folder)

    try:
        data = request.urlopen(url)
    except Exception as e:
        print(e)
        print("Failed to download the dataset.")
        print(f"Please download the dataset manually and put it under {folder}.")
        exit(1)

    if name is None:
        filename = url.rpartition("/")[2]
    else:
        filename = name
    path = osp.join(folder, filename)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    self_weight = torch.full((num_nodes,), fill_value, dtype=edge_weight.dtype).to(edge_weight.device)
    loop_index = torch.arange(0, N, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_weight = torch.cat([edge_weight, self_weight])
    return edge_index, edge_weight


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=device)
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    row, col = edge_index[0], edge_index[1]
    mask = row != col

    loop_index = torch.arange(0, N, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    inv_mask = ~mask

    loop_weight = torch.full((N,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device)
    remaining_edge_weight = edge_weight[inv_mask]
    if remaining_edge_weight.numel() > 0:
        loop_weight[row[inv_mask]] = remaining_edge_weight
    edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    return edge_index, edge_weight


def row_normalization(num_nodes, edge_index, edge_weight=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    row_sum = spmm_scatter(edge_index, edge_weight, torch.ones(num_nodes, 1).to(device))
    row_sum_inv = row_sum.pow(-1).view(-1)
    return edge_weight * row_sum_inv[edge_index[0]]


def symmetric_normalization(num_nodes, edge_index, edge_weight=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    row_sum = spmm_scatter(edge_index, edge_weight, torch.ones(num_nodes, 1).to(device)).view(-1)
    row_sum_inv_sqrt = row_sum.pow(-0.5)
    row_sum_inv_sqrt[row_sum_inv_sqrt == float("inf")] = 0
    return row_sum_inv_sqrt[edge_index[1]] * edge_weight * row_sum_inv_sqrt[edge_index[0]]


def spmm_scatter(indices, values, b):
    r"""
    Args:
        indices : Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        b : Tensor, shape=(N, )
    """
    output = b.index_select(0, indices[1]) * values.unsqueeze(-1)
    output = torch.zeros_like(b).scatter_add_(0, indices[0].unsqueeze(-1).expand_as(output), output)
    return output


def spmm_adj(indices, values, x, num_nodes=None):
    if num_nodes is None:
        num_nodes = x.shape[0]
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=(num_nodes, num_nodes))
    return torch.spmm(adj, x)


fast_spmm = None
_cache = dict()


def initialize_spmm(args):
    if hasattr(args, "fast_spmm") and args.fast_spmm is True:
        try:
            from cogdl.layers.spmm_layer import csrspmm

            global fast_spmm
            fast_spmm = csrspmm
            print("Using fast-spmm to speed up training")
        except Exception as e:
            print(e)


def spmm(edge_index, edge_weight, x, num_nodes=None):
    r"""
    Args:
        edge_index : Tensor, shape=(2, E)
        edge_weight : Tensor, shape=(E,)
        x : Tensor, shape=(N, )
        num_nodes : Optional[int]
    """
    if fast_spmm is not None and str(x.device) != "cpu":
        (colptr, row_indices, csr_data, rowptr, col_indices, csc_data) = get_csr_ind(
            edge_index, edge_weight, x.shape[0]
        )
        x = fast_spmm(rowptr, col_indices, colptr, row_indices, x, csr_data, csc_data)
    elif edge_weight.requires_grad:
        x = spmm_scatter(edge_index, edge_weight, x)
    else:
        x = spmm_adj(edge_index, edge_weight, x, num_nodes)
    return x


def get_csr_ind(edge_index, edge_weight=None, num_nodes=None):
    flag = str(edge_index.shape[1])
    if flag not in _cache:
        if num_nodes is None:
            num_nodes = torch.max(edge_index) + 1
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        _cache[flag] = get_csr_from_edge_index(edge_index, edge_weight, size=(num_nodes, num_nodes))
    cache = _cache[flag]
    colptr = cache["colptr"]
    row_indices = cache["row_indices"]
    csr_data = cache["csr_data"]
    rowptr = cache["rowptr"]
    col_indices = cache["col_indices"]
    csc_data = cache["csc_data"]
    return colptr, row_indices, csr_data, rowptr, col_indices, csc_data


def get_csr_from_edge_index(edge_index, edge_attr, size):
    device = edge_index.device
    _edge_index = edge_index.cpu().numpy()
    _edge_attr = edge_attr.cpu().numpy()
    num_nodes = size[0]

    adj = sp.csr_matrix((_edge_attr, (_edge_index[0], _edge_index[1])), shape=(num_nodes, num_nodes))
    colptr = torch.as_tensor(adj.indptr, dtype=torch.int32).to(device)
    row_indices = torch.as_tensor(adj.indices, dtype=torch.int32).to(device)
    csr_data = torch.as_tensor(adj.data, dtype=torch.float).to(device)
    adj = adj.tocsc()
    rowptr = torch.as_tensor(adj.indptr, dtype=torch.int32).to(device)
    col_indices = torch.as_tensor(adj.indices, dtype=torch.int32).to(device)
    csc_data = torch.as_tensor(adj.data, dtype=torch.float).to(device)
    cache = {
        "colptr": colptr,
        "row_indices": row_indices,
        "csr_data": csr_data,
        "rowptr": rowptr,
        "col_indices": col_indices,
        "csc_data": csc_data,
    }
    return cache


def coo2csr(edge_index, edge_attr, num_nodes=None):
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    device = edge_index[0].device
    sorted_index = torch.argsort(edge_index[0])
    sorted_index = sorted_index.long()
    edge_index = edge_index[:, sorted_index]
    edge_attr = edge_attr[sorted_index]
    indices = edge_index[1]

    row = edge_index[0]
    indptr = torch.zeros(num_nodes + 1, dtype=torch.int32, device=device)
    elements, counts = torch.unique(row, return_counts=True)
    elements = elements.long() + 1
    indptr[elements] = counts.to(indptr.dtype)
    indptr = indptr.cumsum(dim=0)
    return indptr.int(), indices.int(), edge_attr


def coo2csc(edge_index, edge_attr):
    edge_index = torch.stack([edge_index[1], edge_index[0]])
    return coo2csr(edge_index, edge_attr)


# def naive_spspmm(edge_index1, value1, edge_index2, value2, shape):
#     if isinstance(edge_index1, torch.Tensor):
#         row1, col1 = edge_index1.cpu().numpy()
#         value1 = value1.cpu().numpy()
#     else:
#         row1, col1 = edge_index1
#     if isinstance(edge_index2, torch.Tensor):
#         row2, col2 = edge_index2.cpu().numpy()
#         value2 = value2.cpu().numpy()
#     else:
#         row2, col2 = edge_index2
#     adj1 = sp.csr_matrix((value1, (row1, col1)), shape=shape)
#     adj2 = sp.csc_matrix((value2, (row2, col2)), shape=shape)
#     indptr1 = adj1.indptr
#     indices1 = adj1.indices
#
#     indptr2 = adj2.indptr
#     indices2 = adj2.indices
#
#     result = _naive_spspmm(indptr1, indices1, indptr2, indices2)
#
#
# @numba.njit
# def _naive_spspmm(indptr_row, indices_row, indptr_col, indices_col):


def get_degrees(indices, num_nodes=None):
    device = indices.device
    values = torch.ones(indices.shape[1]).to(device)
    if num_nodes is None:
        num_nodes = torch.max(values) + 1
    b = torch.ones((num_nodes, 1)).to(device)
    degrees = spmm_scatter(indices, values, b).view(-1)
    return degrees


def edge_softmax(indices, values, shape):
    """
    Args:
        indices: Tensor, shape=(2, E)
        values: Tensor, shape=(N,)
        shape: tuple(int, int)

    Returns:
        Softmax values of edge values for nodes
    """
    values = torch.exp(values)
    node_sum = spmm_scatter(indices, values, torch.ones(shape[0], 1).to(values.device)).squeeze()
    softmax_values = values / node_sum[indices[0]]
    return softmax_values


def mul_edge_softmax(indices, values, shape):
    """
    Args:
        indices: Tensor, shape=(2, E)
        values: Tensor, shape=(E, d)
        shape: tuple(int, int)

    Returns:
        Softmax values of multi-dimension edge values for nodes
    """
    device = values.device
    values = torch.exp(values)
    output = torch.zeros(shape[0], values.shape[1]).to(device)
    output = output.scatter_add_(0, indices[0].unsqueeze(-1).expand_as(values), values)
    softmax_values = values / (output[indices[0]] + 1e-8)
    softmax_values[torch.isnan(softmax_values)] = 0
    return softmax_values


def remove_self_loops(indices, values=None):
    mask = indices[0] != indices[1]
    indices = indices[:, mask]
    if values is not None:
        values = values[mask]
    return indices, values


def filter_adj(row, col, edge_attr, mask):
    return torch.stack([row[mask], col[mask]]), None if edge_attr is None else edge_attr[mask]


def dropout_adj(
    edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.5, renorm: bool = True
):
    if drop_rate < 0.0 or drop_rate > 1.0:
        raise ValueError("Dropout probability has to be between 0 and 1, " "but got {}".format(drop_rate))

    num_nodes = int(torch.max(edge_index)) + 1
    mask = edge_index.new_full((edge_index.size(1),), 1 - drop_rate, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)
    edge_index, edge_weight = filter_adj(edge_index[0], edge_index[1], edge_weight, mask)
    if renorm:
        edge_weight = symmetric_normalization(num_nodes, edge_index)
    return edge_index, edge_weight


def coalesce(row, col, value=None):
    bigger = ((row[1:] - row[:-1]) >= 0).all()
    if not bigger:
        edge_index = torch.stack([row, col]).T
        sort_value, sort_index = torch.sort(edge_index, dim=0)
        sort_index = sort_index[:, 0]
        edge_index = edge_index[sort_index].t()
        row, col = edge_index
    num = col.shape[0] + 1
    idx = torch.full((num,), -1, dtype=torch.float)
    idx[1:] = row * num + col
    mask = idx[1:] > idx[:-1]

    if mask.all():
        return row, col.value
    row = row[mask]
    if value is not None:
        _value = torch.zeros(row.shape[0], dtype=torch.float).to(row.device)
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
    if num_nodes is None:
        num_nodes = int(torch.max(edge_index)) + 1

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    row, col, _ = coalesce(edge_index[0], edge_index[1], None)
    edge_index = torch.stack([row, col])
    return edge_index


def get_activation(act: str):
    if act == "relu":
        return F.relu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "gelu":
        return F.gelu
    elif act == "prelu":
        return F.prelu
    elif act == "identity":
        return lambda x: x
    else:
        return F.relu


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def batch_sum_pooling(x, batch):
    batch_size = int(torch.max(batch.cpu())) + 1
    # batch_size = len(torch.unique(batch))
    res = torch.zeros(batch_size, x.size(1)).to(x.device)
    return res.scatter_add_(dim=0, index=batch.unsqueeze(-1).expand_as(x), src=x)


def batch_mean_pooling(x, batch):
    values, counts = torch.unique(batch, return_counts=True)
    res = torch.zeros(len(values), x.size(1)).to(x.device)
    res = res.scatter_add_(dim=0, index=batch.unsqueeze(-1).expand_as(x), src=x)
    return res / counts.unsqueeze(-1)


def negative_edge_sampling(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    num_neg_samples: Optional[int] = None,
    undirected: bool = False,
):
    if num_nodes is None:
        num_nodes = len(torch.unique(edge_index))
    if num_neg_samples is None:
        num_neg_samples = edge_index.shape[1]

    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

    row, col = edge_index
    unique_pair = row * num_nodes + col

    num_samples = int(num_neg_samples * abs(1 / (1 - 1.1 * edge_index.size(1) / size)))
    sample_result = torch.LongTensor(random.sample(range(size), min(num_samples, num_samples)))
    mask = torch.from_numpy(np.isin(sample_result, unique_pair.to("cpu"))).to(torch.bool)
    selected = sample_result[~mask][:num_neg_samples].to(edge_index.device)

    row = selected // num_nodes
    col = selected % num_nodes
    return torch.stack([row, col]).long()


def tabulate_results(results_dict):
    # Average for different seeds
    tab_data = []
    for variant in results_dict:
        results = np.array([list(res.values()) for res in results_dict[variant]])
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(
                        np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist(),
                    ),
                )
            )
        )
    return tab_data


def print_result(results, datasets, model_name):
    table_header = ["Variants"] + list(results[0].keys())

    results_dict = defaultdict(list)
    num_datasets = len(datasets)
    num_seed = len(results) // num_datasets
    for i, res in enumerate(results):
        results_dict[(model_name, datasets[i // num_seed])].append(res)
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=table_header, tablefmt="github"))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    args = build_args_from_dict({"a": 1, "b": 2})
    print(args.a, args.b)
