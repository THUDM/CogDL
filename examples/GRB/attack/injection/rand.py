import random
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from ..base import InjectionAttack
from cogdl.utils.grb_utils import eval_acc, feat_preprocess, adj_preprocess, getGraph, getGRBGraph


class RAND(InjectionAttack):
    r"""

    Description
    -----------
    Simple random graph injection attack.

    Parameters
    ----------
    n_inject_max : int
        Maximum number of injected nodes.
    n_edge_max : int
        Maximum number of edges of injected nodes.
    feat_lim_min : float
        Minimum limit of features.
    feat_lim_max : float
        Maximum limit of features.
    loss : func of torch.nn.functional, optional
        Loss function compatible with ``torch.nn.functional``. Default: ``F.cross_entropy``.
    eval_metric : func of grb_utils, optional
        Evaluation metric. Default: ``eval_acc``.
    device : str, optional
        Device used to host data. Default: ``cpu``.
    verbose : bool, optional
        Whether to display logs. Default: ``True``.

    """

    def __init__(
        self,
        n_inject_max,
        n_edge_max,
        feat_lim_min,
        feat_lim_max,
        loss=F.cross_entropy,
        eval_metric=eval_acc,
        device="cpu",
        verbose=True,
    ):
        self.device = device
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose

    def attack(self, model, graph, adj_norm_func):
        r"""

        Description
        -----------
        Attack process consists of injection and feature update.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        graph : cogdl.data.Graph
            Graph to attcak.
        adj_norm_func : func of utils.normalize
            Function that normalizes adjacency matrix.

        Returns
        -------
        out_graph : cogdl.data.Graph
            Graph attacked.

        """
        time_start = time.time()
        # adj, features = getGRBGraph(graph)
        adj = graph.to_scipy_csr()
        target_mask = graph.test_mask
        features = graph.x
        model.to(self.device)
        n_total, n_feat = features.shape
        features = feat_preprocess(features=features, device=self.device)
        adj_tensor = adj_preprocess(adj=adj, adj_norm_func=adj_norm_func, device=self.device)
        pred_origin = model(getGraph(adj_tensor, features, device=self.device))
        labels_origin = torch.argmax(pred_origin, dim=1)
        adj_attack = self.injection(adj=adj, n_inject=self.n_inject_max, n_node=n_total, target_mask=target_mask)

        features_attack = np.zeros((self.n_inject_max, n_feat))
        features_attack = self.update_features(
            model=model,
            adj_attack=adj_attack,
            features_origin=features,
            features_attack=features_attack,
            labels_origin=labels_origin,
            target_mask=target_mask,
            adj_norm_func=adj_norm_func,
        )
        out_features = torch.cat((features, features_attack), 0)
        time_end = time.time()
        if self.verbose:
            print("Attack runtime: {:.4f}.".format(time_end - time_start))

        out_graph = getGraph(adj_attack, out_features, graph.y, device=self.device)
        return out_graph

    def injection(self, adj, n_inject, n_node, target_mask):
        r"""

        Description
        -----------
        Randomly inject nodes to target nodes.

        Parameters
        ----------
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        n_inject : int
            Number of injection.
        n_node : int
            Number of all nodes.
        target_mask : torch.Tensor
            Mask of attack target nodes in form of ``N * 1`` torch bool tensor.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.

        """

        test_index = torch.where(target_mask)[0].cpu()
        n_test = test_index.shape[0]
        new_edges_x = []
        new_edges_y = []
        new_data = []
        for i in range(n_inject):
            islinked = np.zeros(n_test)
            for j in range(self.n_edge_max):
                x = i + n_node

                yy = random.randint(0, n_test - 1)
                while islinked[yy] > 0:
                    yy = random.randint(0, n_test - 1)

                y = test_index[yy]
                islinked[yy] = 1
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])

        add1 = sp.csr_matrix((n_inject, n_node))
        add2 = sp.csr_matrix((n_node + n_inject, n_inject))
        adj_attack = sp.vstack([adj, add1])
        adj_attack = sp.hstack([adj_attack, add2])
        adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
        adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
        adj_attack.data = np.hstack([adj_attack.data, new_data])

        return adj_attack

    def update_features(
        self,
        model,
        adj_attack,
        features_origin,
        features_attack,
        labels_origin,
        target_mask,
        feat_norm=None,
        adj_norm_func=None,
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
        target_mask : torch.Tensor
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

        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max
        n_total = features_origin.shape[0]

        adj_attacked = adj_preprocess(adj=adj_attack, adj_norm_func=adj_norm_func, device=self.device)
        features_attack = np.random.normal(
            loc=0, scale=feat_lim_max, size=(self.n_inject_max, features_origin.shape[1])
        )
        features_attack = np.clip(features_attack, feat_lim_min, feat_lim_max)
        features_attack = feat_preprocess(features=features_attack, feat_norm=feat_norm, device=self.device)
        model.eval()

        features_concat = torch.cat((features_origin, features_attack), dim=0)
        pred = model(getGraph(adj_attacked, features_concat, device=self.device))
        pred_loss = -self.loss(pred[:n_total][target_mask], labels_origin[target_mask]).to(self.device)

        test_acc = self.eval_metric(pred[:n_total][target_mask], labels_origin[target_mask])

        if self.verbose:
            print("Loss: {:.4f}, Surrogate test acc: {:.4f}".format(pred_loss, test_acc))

        return features_attack
