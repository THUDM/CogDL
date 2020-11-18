import itertools
import random
from collections import defaultdict

import numpy as np
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


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    if fill_value is None:
        fill_value = 1

    N = num_nodes
    self_weight = torch.full((num_nodes, ), fill_value, dtype=edge_weight.dtype).to(edge_weight.device)
    loop_index = torch.arange(0, N, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_weight = torch.cat([edge_weight, self_weight])
    return edge_index, edge_weight

def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
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

    loop_weight = torch.full((N, ), fill_value, dtype=edge_weight.dtype,
                                device=edge_weight.device)
    remaining_edge_weight = edge_weight[inv_mask]
    if remaining_edge_weight.numel() > 0:
        loop_weight[row[inv_mask]] = remaining_edge_weight
    edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    return edge_index, edge_weight


def row_normalization(num_nodes, edge_index, edge_weight=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    row_sum = spmm(edge_index, edge_weight, torch.ones(num_nodes, 1).to(device))
    row_sum_inv = row_sum.pow(-1).view(-1)
    return edge_weight * row_sum_inv[edge_index[0]]


def symmetric_normalization(num_nodes, edge_index, edge_weight=None):
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    row_sum = spmm(edge_index, edge_weight, torch.ones(num_nodes, 1).to(device)).view(-1)
    row_sum_inv_sqrt = row_sum.pow(-0.5)
    row_sum_inv_sqrt[row_sum_inv_sqrt == float('inf')] = 0
    return row_sum_inv_sqrt[edge_index[1]] * edge_weight * row_sum_inv_sqrt[edge_index[0]]


def spmm(indices, values, b):
    r"""
    Args:
        indices : Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        shape : tuple(int ,int)
        b : Tensor, shape=(N, )
    """
    output = b.index_select(0, indices[1]) * values.unsqueeze(-1)
    output = torch.zeros_like(b).scatter_add_(0, indices[0].unsqueeze(-1).expand_as(output), output)
    return output


def spmm_adj(indices, values, shape, b):
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=shape)
    return torch.spmm(adj, b)


def get_degrees(indices, num_nodes=None):
    device = indices.device
    values = torch.ones(indices.shape[1]).to(device)
    if num_nodes is None:
        num_nodes = torch.max(values) + 1
    b = torch.ones((num_nodes, 1)).to(device)
    degrees = spmm(indices, values, b).view(-1)
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
    node_sum = spmm(indices, values, torch.ones(shape[0], 1).to(values.device)).squeeze()
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
    softmax_values = values / output[indices[0]]
    return softmax_values


def remove_self_loops(indices):
    mask = indices[0] != indices[1]
    indices = indices[:, mask]
    return indices, mask


def get_activation(act):
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
    else:
        return F.relu


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def batch_sum_pooling(x, batch):
    batch_size = torch.max(batch.cpu())+1
    # batch_size = len(torch.unique(batch))
    res = torch.zeros(batch_size, x.size(1)).to(x.device)
    return res.scatter_add_(dim=0, index=batch.unsqueeze(-1).expand_as(x), src=x)


def batch_mean_pooling(x, batch):
    values, counts = torch.unique(batch, return_counts=True)
    res = torch.zeros(len(values), x.size(1)).to(x.device)
    res = res.scatter_add_(dim=0, index=batch.unsqueeze(-1).expand_as(x), src=x)
    return res / counts.unsqueeze(-1)

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
        results_dict[(model_name, datasets[i//num_seed])].append(res)
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=table_header, tablefmt="github"))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    args = build_args_from_dict({'a': 1, 'b': 2})
    print(args.a, args.b)
