import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
from tqdm.auto import tqdm

from ..base import ModificationAttack
from cogdl.data import Graph
from cogdl.utils.grb_utils import getGraph


class STACK(ModificationAttack):
    """
    STACK.
    """

    def __init__(self, n_edge_mod, allow_isolate=True, device="cpu", verbose=True):
        self.n_edge_mod = n_edge_mod
        self.allow_isolate = allow_isolate
        self.device = device
        self.verbose = verbose

    def attack(self, graph: Graph, **kwargs):
        adj_attack = self.modification(graph.to_scipy_csr(), graph.test_nid.cpu())

        return getGraph(adj_attack, graph.x, graph.y, device=self.device)

    def modification(self, adj, index_target):
        adj_attack = adj.copy()
        degs = adj_attack.getnnz(axis=1)
        adj_ = adj + sp.eye(adj.shape[0])
        eigen_vals, eigen_vecs = spl.eigh(adj_.toarray(), np.diag(adj_.getnnz(axis=1)))
        index_i, index_j = index_target[adj[index_target].nonzero()[0]], adj[index_target].nonzero()[1]
        edges_target = np.column_stack([index_i, index_j])

        flip_indicator = 1 - 2 * np.array(adj[tuple(edges_target.T)])[0]
        eigen_scores = np.zeros(len(edges_target))
        sub_org = np.sqrt(np.sum(eigen_vals ** 2))
        for x in range(len(edges_target)):
            i, j = edges_target[x]
            vals_est = eigen_vals + flip_indicator[x] * (
                2 * eigen_vecs[i] * eigen_vecs[j] - eigen_vals * (eigen_vecs[i] ** 2 + eigen_vecs[j] ** 2)
            )
            loss_ij = np.abs(sub_org - np.sqrt(np.sum(vals_est ** 2)))
            eigen_scores[x] = loss_ij
        struct_scores = np.expand_dims(eigen_scores, 1)
        flip_edges_idx = np.argsort(struct_scores, axis=0)[::-1]
        flip_edges = edges_target[flip_edges_idx].squeeze()

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
            print("STACK attack finished. {:d} edges were flipped.".format(n_edge_flip))

        return adj_attack


# def sum_of_powers(x, power):
#     n = x.shape[0]
#     sum_powers = np.zeros((power, n))
#     for i, i_power in enumerate(range(1, power + 1)):
#         sum_powers[i] = np.power(x, i_power)

#     return sum_powers.sum(0)
