import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from attack.base import ModificationAttack, EarlyStop
from cogdl.data import Graph
from cogdl.utils.grb_utils import eval_acc, feat_preprocess, adj_preprocess, getGraph, getGRBGraph


class PGD(ModificationAttack):
    def __init__(
        self,
        epsilon,
        n_epoch,
        n_node_mod,
        n_edge_mod,
        feat_lim_min,
        feat_lim_max,
        allow_isolate=False,
        loss=F.cross_entropy,
        eval_metric=eval_acc,
        early_stop=None,
        early_stop_patience=1000,
        early_stop_epsilon=1e-5,
        device="cpu",
        verbose=True,
    ):
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.n_node_mod = n_node_mod
        self.n_edge_mod = n_edge_mod
        self.allow_isolate = allow_isolate
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.device = device
        self.verbose = verbose

        # Early stop
        if early_stop:
            if isinstance(early_stop, EarlyStop):
                self.early_stop = early_stop
            else:
                self.early_stop = EarlyStop(patience=early_stop_patience, epsilon=early_stop_epsilon)
        else:
            self.early_stop = None

    def attack(self, model, graph: Graph, feat_norm=None, adj_norm_func=None):
        time_start = time.time()
        model.to(self.device)
        # adj, features = getGRBGraph(graph)
        adj = graph.to_scipy_csr()
        index_target = graph.test_nid.cpu()
        features = graph.x.clone().detach()
        features = feat_preprocess(features=features, feat_norm=feat_norm, device=self.device)
        adj_tensor = adj_preprocess(adj=adj, adj_norm_func=adj_norm_func, device=self.device)
        pred_origin = model(getGraph(adj_tensor, features, device=self.device))
        labels_origin = torch.argmax(pred_origin, dim=1)
        adj_attack = self.modification(adj, index_target)

        features_attack = self.update_features(
            model=model,
            adj_attack=adj_attack,
            features_origin=features,
            labels_origin=labels_origin,
            index_target=index_target,
        )

        time_end = time.time()
        if self.verbose:
            print("Attack runtime: {:.4f}.".format(time_end - time_start))

        return getGraph(adj_attack, features_attack, graph.y, device=self.device)

    def modification(self, adj, index_target):
        # if type(adj) == torch.Tensor:
        #     adj_attack = adj.clone().to_dense()
        # else:
        #     adj_attack = adj.todense()
        #     adj_attack = torch.FloatTensor(adj_attack)
        # degs = adj_attack.sum(dim=1)
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
        return adj_attack

    def update_features(
        self, model, adj_attack, features_origin, labels_origin, index_target, feat_norm=None, adj_norm_func=None
    ):
        r"""

        Description
        -----------
        Update features of injected nodes.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj_attack :  scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.
        features_origin : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        features_attack : torch.FloatTensor
            Features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.
        labels_origin : torch.LongTensor
            Labels of target nodes originally predicted by the model.
        index_target : torch.Tensor
            Mask of target nodes in form of ``N * 1`` torch bool tensor.
        feat_norm : str, optional
            Type of feature normalization, ['arctan', 'tanh']. Default: ``None``.
        adj_norm_func : func of utils.normalize, optional
            Function that normalizes adjacency matrix. Default: ``None``.

        Returns
        -------
        features_attack : torch.FloatTensor
            Updated features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.

        """

        epsilon = self.epsilon
        n_epoch = self.n_epoch
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        features_attack = feat_preprocess(features=features_origin, feat_norm=feat_norm, device=self.device)
        adj_attacked_tensor = adj_preprocess(adj=adj_attack, adj_norm_func=adj_norm_func, device=self.device)
        index_mod = np.random.choice(index_target, self.n_node_mod)
        model.eval()
        epoch_bar = tqdm(range(n_epoch), disable=not self.verbose)
        for i in epoch_bar:
            features_attack.requires_grad_(True)
            features_attack.retain_grad()
            pred = model(getGraph(adj_attacked_tensor, features_attack, device=self.device))
            pred_loss = self.loss(pred[index_target], labels_origin[index_target]).to(self.device)

            model.zero_grad()
            pred_loss.backward()
            grad = features_attack.grad.data
            features_attack = features_attack.detach()
            features_attack[index_mod] = features_attack.clone()[index_mod] + epsilon * grad.sign()[index_mod]
            features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)

            test_score = self.eval_metric(pred[index_target], labels_origin[index_target])

            if self.early_stop:
                self.early_stop(test_score)
                if self.early_stop.stop:
                    if self.verbose:
                        print("Attack early stopped.Surrogate test score: {:.4f}".format(test_score))
                    self.early_stop = EarlyStop()

                    return features_attack
            if self.verbose:
                epoch_bar.set_description(
                    "Epoch {}, Loss: {:.4f}, Surrogate test score: {:.4f}".format(i, pred_loss, test_score)
                )
        if self.verbose:
            print("Surrogate test score: {:.4f}".format(test_score))

        return features_attack
