import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from ..base import ModificationAttack
from cogdl.data import Graph
from cogdl.utils.grb_utils import getGraph


class FLIP(ModificationAttack):
    """
    FLIP, degree, betweenness, eigen.
    """

    def __init__(self, n_edge_mod, flip_type="deg", mode="descend", allow_isolate=True, device="cpu", verbose=True):
        self.n_edge_mod = n_edge_mod
        self.flip_type = flip_type
        self.mode = mode
        self.allow_isolate = allow_isolate
        self.device = device
        self.verbose = verbose

    def attack(self, graph: Graph, **kwargs):
        adj_attack = self.modification(
            graph.to_scipy_csr(), graph.test_nid.cpu(), flip_type=self.flip_type, mode=self.mode, **kwargs
        )

        return getGraph(adj_attack, graph.x, graph.y, device=self.device)

    def modification(self, adj, index_target, flip_type="deg", saved=None, mode="descend"):
        adj_attack = adj.copy()
        degs = adj_attack.getnnz(axis=1)
        if flip_type == "deg":
            flip_edges = get_degree_flips_edges(adj, index_target, mode=mode)
        elif flip_type == "bet":
            flip_edges = betweenness_flips(adj, index_target, saved_bets=saved, mode=mode)
        elif flip_type == "eigen":
            flip_edges = eigen_flips(adj, index_target, saved_eigens=saved, mode=mode)
        else:
            raise NotImplementedError
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
            print("FLIP attack finished. {:d} edges were flipped.".format(n_edge_flip))

        return adj_attack


def get_degree_flips_edges(adj, index_target, mode="descend"):
    degs = adj.getnnz(axis=1)
    index_i, index_j = index_target[adj[index_target].nonzero()[0]], adj[index_target].nonzero()[1]
    deg_score = degs[index_i] + degs[index_j]
    if mode == "ascend":
        deg_score = deg_score
    elif mode == "descend":
        deg_score = -deg_score
    else:
        raise NotImplementedError
    edges_target = np.column_stack([index_i, index_j])
    flip_edges_idx = np.argsort(deg_score, axis=0)
    flip_edges = edges_target[flip_edges_idx].squeeze()

    return flip_edges


def betweenness_flips(adj, index_target, saved_bets=None, mode="descend"):
    if saved_bets is None:
        g = nx.from_scipy_sparse_matrix(adj)
        bets = nx.betweenness_centrality(g)
        bets = np.array(list(bets.values()))
    else:
        bets = saved_bets
    index_i, index_j = index_target[adj[index_target].nonzero()[0]], adj[index_target].nonzero()[1]
    bet_score = bets[index_i] + bets[index_j]
    if mode == "ascend":
        bet_score = bet_score
    elif mode == "descend":
        bet_score = -bet_score
    else:
        raise NotImplementedError
    edges_target = np.column_stack([index_i, index_j])
    flip_edges_idx = np.argsort(bet_score, axis=0)
    flip_edges = edges_target[flip_edges_idx].squeeze()

    return flip_edges


def eigen_flips(adj, index_target, saved_eigens=None, mode="descend"):
    if saved_eigens is None:
        g = nx.from_scipy_sparse_matrix(adj)
        eigens = nx.eigenvector_centrality(g)
        eigens = np.array(list(eigens.values()))
    else:
        eigens = saved_eigens
    index_i, index_j = index_target[adj[index_target].nonzero()[0]], adj[index_target].nonzero()[1]
    eigen_score = eigens[index_i] + eigens[index_j]
    if mode == "ascend":
        eigen_score = eigen_score
    elif mode == "descend":
        eigen_score = -eigen_score
    else:
        raise NotImplementedError

    edges_target = np.column_stack([index_i, index_j])
    flip_edges_idx = np.argsort(eigen_score, axis=0)
    flip_edges = edges_target[flip_edges_idx].squeeze()

    return flip_edges
