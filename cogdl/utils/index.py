import torch
from .spmm_utils import spmm


@torch.no_grad()
def homo_index(g, x):
    with g.local_graph():
        g.remove_self_loops()
        neighbors = spmm(g, x)
        deg = g.degrees()
    isolated_nodes = deg == 0
    diff = (x - neighbors).norm(2, dim=-1)
    diff = diff.mean(1)
    diff = diff[~isolated_nodes]
    return torch.mean(diff)


@torch.no_grad()
def mad_index(g, x):
    row, col = g.edge_index
    self_loop = row == col
    mask = ~self_loop
    row = row[mask]
    col = col[mask]

    src, tgt = x[col], x[row]
    sim = (src * tgt).sum(dim=1)
    src_size = src.norm(p=2, dim=1)
    tgt_size = tgt.norm(p=2, dim=1)
    distance = 1 - sim / (src_size * tgt_size)

    N = g.num_nodes

    deg = g.degrees() - 1
    out = torch.zeros((N,), dtype=torch.float, device=x.device)
    out = out.scatter_add_(index=row, dim=0, src=distance)
    deg_inv = deg.pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 1
    dis = out * deg_inv
    dis = dis[dis > 0]
    return torch.mean(dis).item()
