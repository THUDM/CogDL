import numpy as np
from tqdm.auto import tqdm
import copy

from ..base import ModificationAttack
from cogdl.data import Graph
from cogdl.utils.grb_utils import getGraph, updateGraph


class DICE(ModificationAttack):
    """
    DICE (delete internally, connect externally)
    """

    def __init__(self, n_edge_mod, ratio_delete=0.6, allow_isolate=True, device="cpu", verbose=True):
        self.n_edge_mod = n_edge_mod
        self.ratio_delete = ratio_delete
        self.allow_isolate = allow_isolate
        self.device = device
        self.verbose = verbose

    def attack(self, graph: Graph):
        if graph.test_mask is None or len(graph.test_nid) == 0:
            print("index_target is None.")
            exit(1)
        adj_attack = self.modification(graph.to_scipy_csr(), graph.test_nid.cpu(), graph.y)
        # graph = copy.deepcopy(graph)
        # updateGraph(graph, adj_attack, graph.x)
        # return graph
        return getGraph(adj_attack, graph.x, graph.y, device=self.device)

    def modification(self, adj, index_target, labels):
        adj_attack = adj.tolil()
        degrees = adj_attack.getnnz(axis=1)

        # delete internally
        print("Delete internally......")
        n_delete = int(np.floor(self.n_edge_mod * self.ratio_delete))
        index_i, index_j = index_target[adj_attack[index_target].nonzero()[0]], adj_attack[index_target].nonzero()[1]
        target_index_pair = []
        for index in tqdm(zip(index_i, index_j), total=len(index_i)):
            if index[0] != index[1] and labels[index[0]] == labels[index[1]]:
                if self.allow_isolate:
                    # if index[::-1] not in target_index_pair:
                    target_index_pair.append(index)
                else:
                    if degrees[index[0]] > 1 and degrees[index[1]] > 1:
                        # if index[::-1] not in target_index_pair:
                        target_index_pair.append(index)
                        degrees[index[0]] -= 1
                        degrees[index[1]] -= 1

        index_delete = np.random.permutation(target_index_pair)[:n_delete]
        if index_delete != []:
            adj_attack[index_delete[:, 0], index_delete[:, 1]] = 0
            adj_attack[index_delete[:, 1], index_delete[:, 0]] = 0

        # connect externally
        print("Connect externally......")
        n_connect = self.n_edge_mod - n_delete
        index_i, index_j = index_target, np.arange(adj_attack.shape[1])
        target_index_pair = []
        for index in tqdm(zip(index_i, index_j), total=len(index_i)):
            if index[0] != index[1] and labels[index[0]] != labels[index[1]] and adj_attack[index[0], index[1]] == 0:
                # if index[::-1] not in target_index_pair:
                target_index_pair.append(index)

        index_connect = np.random.permutation(target_index_pair)[:n_connect]
        adj_attack[index_connect[:, 0], index_connect[:, 1]] = 1
        adj_attack[index_connect[:, 1], index_connect[:, 0]] = 1
        adj_attack = adj_attack.tocsr()
        adj_attack.eliminate_zeros()

        if self.verbose:
            print(
                "DICE attack finished. {:d} edges were removed, {:d} edges were connected.".format(n_delete, n_connect)
            )

        return adj_attack
