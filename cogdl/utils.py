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

if __name__ == "__main__":
    args = build_args_from_dict({'a': 1, 'b': 2})
    print(args.a, args.b)
