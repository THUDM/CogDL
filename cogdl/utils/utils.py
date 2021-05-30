import errno
import itertools
import os
import os.path as osp
import random
import shutil
from collections import defaultdict
from typing import Optional, Tuple
from urllib import request

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tabulate import tabulate

from cogdl.operators.sample import coo2csr_cpu, coo2csr_cpu_index


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
        val = torch.ones(row.shape[0]).to(device)
    row_sum = spmm_scatter(row, col, val, torch.ones(num_nodes, 1).to(device))
    row_sum_inv = row_sum.pow(-1).view(-1)
    row_sum_inv[torch.isinf(row_sum_inv)] = 0
    return val * row_sum_inv[row]


def symmetric_normalization(num_nodes, row, col, val=None):
    device = row.device
    if val is None:
        val = torch.ones(row.shape[0]).to(device)
    row_sum = spmm_scatter(row, col, val, torch.ones(num_nodes, 1).to(device)).view(-1)
    row_sum_inv_sqrt = row_sum.pow(-0.5)
    row_sum_inv_sqrt[row_sum_inv_sqrt == float("inf")] = 0
    return row_sum_inv_sqrt[col] * val * row_sum_inv_sqrt[row]


def spmm_scatter(row, col, values, b):
    r"""
    Args:
        indices : Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        b : Tensor, shape=(N, )
    """
    output = b.index_select(0, col) * values.unsqueeze(-1)
    output = torch.zeros_like(b).scatter_add_(0, row.unsqueeze(-1).expand_as(output), output)
    return output


def spmm_adj(indices, values, x, num_nodes=None):
    if num_nodes is None:
        num_nodes = x.shape[0]
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=(num_nodes, num_nodes))
    return torch.spmm(adj, x)


CONFIGS = {"fast_spmm": None, "csrmhspmm": None, "csr_edge_softmax": None, "spmm_flag": False, "mh_spmm_flag": False}


def init_operator_configs(args=None):
    if args is not None and args.cpu:
        CONFIGS["fast_spmm"] = None
        CONFIGS["csrmhspmm"] = None
        CONFIGS["csr_edge_softmax"] = None
        return

    if CONFIGS["spmm_flag"] or CONFIGS["mh_spmm_flag"]:
        return
    if args is None:
        args = build_args_from_dict({"fast-spmm": True, "cpu": not torch.cuda.is_available()})
    initialize_spmm(args)
    initialize_edge_softmax(args)


def initialize_spmm(args):
    CONFIGS["spmm_flag"] = True
    if hasattr(args, "fast_spmm") and args.fast_spmm is True and not args.cpu:
        try:
            from cogdl.operators.spmm import csrspmm

            CONFIGS["fast_spmm"] = csrspmm
            print("Using fast-spmm to speed up training")
        except Exception:
            print("Failed to load fast version of SpMM, use torch.scatter_add instead.")


def initialize_edge_softmax(args):
    CONFIGS["mh_spmm_flag"] = True
    if torch.cuda.is_available() and not args.cpu:
        from cogdl.operators.edge_softmax import csr_edge_softmax
        from cogdl.operators.mhspmm import csrmhspmm

        CONFIGS["csrmhspmm"] = csrmhspmm
        CONFIGS["csr_edge_softmax"] = csr_edge_softmax


def check_fast_spmm():
    return CONFIGS["fast_spmm"] is not None


def check_mh_spmm():
    return CONFIGS["csrmhspmm"] is not None


def check_edge_softmax():
    return CONFIGS["csr_edge_softmax"] is not None


def spmm(graph, x):
    if graph.out_norm is not None:
        x = graph.out_norm * x

    fast_spmm = CONFIGS["fast_spmm"]
    if fast_spmm is not None and str(x.device) != "cpu":
        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.edge_weight
        x = fast_spmm(row_ptr.int(), col_indices.int(), x, csr_data.contiguous(), graph.is_symmetric())
    else:
        row, col = graph.edge_index
        x = spmm_scatter(row, col, graph.edge_weight, x)
    if graph.in_norm is not None:
        x = graph.in_norm * x
    return x


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
        num_nodes = torch.max(torch.stack(row, col)).item() + 1
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


def get_degrees(row, col, num_nodes=None):
    device = row.device
    values = torch.ones(row.shape[0]).to(device)
    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1
    b = torch.ones((num_nodes, 1)).to(device)
    degrees = spmm_scatter(row, col, values, b).view(-1)
    return degrees


def edge_softmax(graph, edge_val):
    """
    Args:
        indices: Tensor, shape=(2, E)
        values: Tensor, shape=(N,)
        shape: tuple(int, int)

    Returns:
        Softmax values of edge values for nodes
    """
    edge_val_max = edge_val.max().item()
    while edge_val_max > 10:
        edge_val -= edge_val / 2
        edge_val_max = edge_val.max().item()

    with graph.local_graph():
        edge_val = torch.exp(edge_val)
        graph.edge_weight = edge_val
        x = torch.ones(graph.num_nodes, 1).to(edge_val.device)
        node_sum = spmm(graph, x).squeeze()
        row = graph.edge_index[0]
        softmax_values = edge_val / node_sum[row]
        return softmax_values


def mul_edge_softmax(graph, edge_val):
    """
    Returns:
        Softmax values of multi-dimension edge values. shape: [E, H]
    """
    csr_edge_softmax = CONFIGS["csr_edge_softmax"]
    if csr_edge_softmax is not None and edge_val.device.type != "cpu":
        val = csr_edge_softmax(graph.row_indptr.int(), edge_val)
        return val
    else:
        val = []
        for i in range(edge_val.shape[1]):
            val.append(edge_softmax(graph, edge_val[:, i]))
        return torch.stack(val).t()


def mh_spmm(graph, attention, h):
    """
        Multi-head spmm
    Args:
        graph: Graph
        attention: torch.Tensor([E, H])
        h: torch.Tensor([N, d])

    Returns:
        torch.Tensor([N, H, d])
    """
    csrmhspmm = CONFIGS["csrmhspmm"]
    return csrmhspmm(graph.row_indptr.int(), graph.col_indices.int(), h, attention)


def remove_self_loops(indices, values=None):
    row, col = indices
    mask = indices[0] != indices[1]
    row = row[mask]
    col = col[mask]
    if values is not None:
        values = values[mask]
    return (row, col), values


def filter_adj(row, col, edge_attr, mask):
    return (row[mask], col[mask]), None if edge_attr is None else edge_attr[mask]


def dropout_adj(
    edge_index: Tuple, edge_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.5, renorm: bool = True
):
    if drop_rate < 0.0 or drop_rate > 1.0:
        raise ValueError("Dropout probability has to be between 0 and 1, " "but got {}".format(drop_rate))

    row, col = edge_index
    num_nodes = int(max(row.max(), col.max())) + 1
    mask = torch.full((row.shape[0],), 1 - drop_rate, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)
    edge_index, edge_weight = filter_adj(row, col, edge_weight, mask)
    if renorm:
        edge_weight = symmetric_normalization(num_nodes, edge_index[0], edge_index[1])
    return edge_index, edge_weight


def coalesce(row, col, value=None):
    row = row.numpy()
    col = col.numpy()
    indices = np.lexsort((col, row))
    row = torch.from_numpy(row[indices])
    col = torch.from_numpy(col[indices])

    num = col.shape[0] + 1
    idx = torch.full((num,), -1, dtype=torch.float)
    idx[1:] = row * num + col
    mask = idx[1:] > idx[:-1]

    if mask.all():
        return row, col, value
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

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    row, col, _ = coalesce(row, col, None)
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


if __name__ == "__main__":
    args = build_args_from_dict({"a": 1, "b": 2})
    print(args.a, args.b)
