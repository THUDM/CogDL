import numpy as np
from tqdm.auto import tqdm

from ..base import ModificationAttack
from cogdl.data import Graph
from cogdl.utils.grb_utils import getGraph


class RAND(ModificationAttack):
    """
    FLIP, degree, betweenness, eigen.
    """

    def __init__(self, n_edge_mod, allow_isolate=True, device="cpu", verbose=True):
        self.n_edge_mod = n_edge_mod
        self.allow_isolate = allow_isolate
        self.device = device
        self.verbose = verbose

    def attack(self, graph: Graph):
        adj_attack = self.modification(graph.to_scipy_csr(), graph.test_nid.cpu())

        return getGraph(adj_attack, graph.x, graph.y, device=self.device)

    def modification(self, adj, index_target):
        adj_attack = adj.copy()
        degs = adj_attack.getnnz(axis=1)

        # Randomly flip edges
        index_i, index_j = index_target[adj_attack[index_target].nonzero()[0]], adj_attack[index_target].nonzero()[1]
        flip_edges = np.random.permutation(np.column_stack([index_i, index_j]))
        n_edge_flip = 0
        for index in tqdm(flip_edges):
            if n_edge_flip >= self.n_edge_mod:
                break
            if adj_attack[index[0], index[1]] == 0:
                adj_attack[index[0], index[1]] = 1
                adj_attack[index[1], index[0]] = 1
                degs[index[0]] += 1
                degs[index[1]] += 1
                n_edge_flip += 1
            else:
                if self.allow_isolate:
                    adj_attack[index[0], index[1]] = 0
                    adj_attack[index[1], index[0]] = 0
                    n_edge_flip += 1
                else:
                    if degs[index[0]] > 1 and degs[index[1]] > 1:
                        adj_attack[index[0], index[1]] = 0
                        adj_attack[index[1], index[0]] = 0
                        degs[index[0]] -= 1
                        degs[index[1]] -= 1
                        n_edge_flip += 1
        adj_attack.eliminate_zeros()
        if self.verbose:
            print("RAND attack finished. {:d} edges were randomly flipped.".format(n_edge_flip))

        return adj_attack
