import torch


class ArgClass(object):
    def __init__(self):
        pass

def build_args_from_dict(dic):
    args = ArgClass()
    for key, value in dic.items():
        args.__setattr__(key, value)
    return args

def add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes):
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
    output = torch.zeros_like(b).scatter_add_(0, indices[0].unsqueeze(-1).repeat(1, b.shape[1]), output)
    return output


def spmm_adj(indices, values, shape, b):
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=shape)
    return torch.spmm(adj, b)


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
    output = output.scatter_add_(0, indices[0].unsqueeze(-1).repeat(1, values.shape[1]), values)
    softmax_values = values / output[indices[0]]
    return softmax_values


def remove_self_loops(indices):
    mask = indices[0] != indices[1]
    indices = indices[:, mask]
    return indices, mask


if __name__ == "__main__":
    args = build_args_from_dict({'a': 1, 'b': 2})
    print(args.a, args.b)
