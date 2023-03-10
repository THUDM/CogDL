from cogdl import function as BF
from .spmm_utils import spmm

from cogdl.backend import BACKEND


def homo_index(g, x):
    def no_grad():
        with g.local_graph():
            g.remove_self_loops()
            neighbors = spmm(g, x)
            deg = g.degrees()
        isolated_nodes = deg == 0
        diff = (x - neighbors).norm(2, dim=-1)
        diff = diff.mean(1)
        diff = diff[BF.logical_not(isolated_nodes)]
        return BF.mean(diff)
    if BACKEND == 'jittor':
        import jittor 
        with jittor.no_grad():
            no_grad()
    elif BACKEND == 'torch':
        import torch
        with torch.no_grad():
            no_grad()


def mad_index(g, x):
    def no_grad():
        row, col = g.edge_index
        self_loop = row == col
        mask = BF.logical_not(self_loop)
        row = row[mask]
        col = col[mask]

        src, tgt = x[col], x[row]
        sim = (src * tgt).sum(dim=1)
        src_size = src.norm(p=2, dim=1)
        tgt_size = tgt.norm(p=2, dim=1)
        distance = 1 - sim / (src_size * tgt_size)

        N = g.num_nodes

        deg = g.degrees() - 1
        out = BF.zeros((N,), dtype=BF.dtype_dict('float'), device=BF.device(x))
        out = BF.scatter_add_(out,index=row, dim=0, src=distance)
        deg_inv = deg.pow(-1)
        deg_inv[BF.isinf(deg_inv)] = 1
        dis = out * deg_inv
        dis = dis[dis > 0]
        return BF.mean(dis).item()
    if BACKEND == 'jittor':
        import jittor 
        with jittor.no_grad():
            no_grad()
    elif BACKEND == 'torch':
        import torch
        with torch.no_grad():
            no_grad()
