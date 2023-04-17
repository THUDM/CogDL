import numba
import numpy as np
import torch
import scipy.sparse as sp
import random

# from cogdl.utils.rwalk import random_walk as c_random_walk


@numba.njit(cache=True, parallel=True)
def random_walk_parallel(start, length, indptr, indices, p=0.0):
    """
    Parameters:
        start : np.array(dtype=np.int32)
        length : int
        indptr : np.array(dtype=np.int32)
        indices : np.array(dtype=np.int32)
        p : float
    Return:
        list(np.array(dtype=np.int32))
    """
    result = [np.zeros(length, dtype=np.int32)] * len(start)
    for i in numba.prange(len(start)):
        result[i] = _random_walk(start[i], length, indptr, indices, p)
    return result


@numba.njit(cache=True)
def random_walk_single(start, length, indptr, indices, p=0.0):
    """
    Parameters:
        start : np.array(dtype=np.int32)
        length : int
        indptr : np.array(dtype=np.int32)
        indices : np.array(dtype=np.int32)
        p : float
    Return:
        list(np.array(dtype=np.int32))
    """
    result = [np.zeros(length, dtype=np.int32)] * len(start)
    for i in numba.prange(len(start)):
        result[i] = _random_walk(start[i], length, indptr, indices, p)
    return result


@numba.njit(cache=True)
def _random_walk(node, length, indptr, indices, p=0.0):
    result = [numba.int32(0)] * length
    result[0] = numba.int32(node)
    i = numba.int32(1)
    _node = node
    _start = indptr[_node] 
    _end = indptr[_node + 1]
    while i < length:
        start = indptr[node] 
        end = indptr[node + 1]
        sample = random.randint(start, end - 1)
        node = indices[sample]
        if np.random.uniform(0, 1) > p:
            result[i] = node
        else:
            sample = random.randint(_start, _end - 1)
            node = indices[sample]
            result[i] = node
        i += 1
    return np.array(result, dtype=np.int32)


class RandomWalker(object):
    def __init__(self, adj=None, num_nodes=None):
        if adj is None:
            self.indptr = None
            self.indices = None
        else:
            if isinstance(adj, torch.Tensor):
                if num_nodes is None:
                    num_nodes = int(torch.max(adj)) + 1
                row, col = adj.cpu().numpy()
                data = np.ones(row.shape[0])
                adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
            adj = adj.tocsr()

            self.indptr = adj.indptr
            self.indices = adj.indices

    def build_up(self, adj, num_nodes):
        if self.indptr is not None:
            return
        if isinstance(adj, torch.Tensor) or isinstance(adj, tuple):
            row, col = adj
            if num_nodes is None:
                num_nodes = int(max(row.max(), col.max())) + 1
            row, col = row.cpu().numpy(), col.cpu().numpy()
            data = np.ones(row.shape[0])
            adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        adj = adj.tocsr()

        self.indptr = adj.indptr
        self.indices = adj.indices

    def walk(self, start, walk_length, restart_p=0.0, parallel=True):
        assert self.indptr is not None, "Please build the adj_list first"
        if isinstance(start, torch.Tensor):
            start = start.cpu().numpy()
        if isinstance(start, list):
            start = np.asarray(start, dtype=np.int32)
        if parallel:
            result = random_walk_parallel(start, walk_length, self.indptr, self.indices, restart_p)
        else:
            result = random_walk_single(start, walk_length, self.indptr, self.indices, restart_p)
        result = np.array(result, dtype=np.int64)
        return result
