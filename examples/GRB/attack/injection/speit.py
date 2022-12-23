import random
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..base import InjectionAttack, EarlyStop
from cogdl.utils.grb_utils import eval_acc, feat_preprocess, adj_preprocess, getGraph, getGRBGraph


class SPEIT(InjectionAttack):
    r"""

    Description
    -----------
    SPEIT graph injection attack, 1st place solution of KDD CUP 2020 Graph Adversarial Attack & Defense.
    (`SPEIT <https://github.com/Stanislas0/KDD_CUP_2020_MLTrack2_SPEIT>`__).

    Parameters
    ----------
    lr : float
        Learning rate of feature optimization process.
    n_epoch : int
        Epoch of perturbations.
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
    inject_mode : str, optional
        Mode of injection. Choose from ``["random", "random-inter", "multi-layer"]``. Default: ``random``.
    early_stop : bool or instance of EarlyStop, optional
        Whether to early stop. Default: ``None``.
    early_stop_patience : int, optional
        Patience of early_stop. Only enabled when ``early_stop is not None``. Default: ``1000``.
    early_stop_epsilon : float, optional
        Tolerance of early_stop. Only enabled when ``early_stop is not None``. Default: ``1e-5``.
    verbose : bool, optional
        Whether to display logs. Default: ``True``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    """

    def __init__(
        self,
        lr,
        n_epoch,
        n_inject_max,
        n_edge_max,
        feat_lim_min,
        feat_lim_max,
        loss=F.cross_entropy,
        eval_metric=eval_acc,
        inject_mode="random",
        early_stop=None,
        early_stop_patience=1000,
        early_stop_epsilon=1e-5,
        verbose=True,
        device="cpu",
    ):
        self.device = device
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.inject_mode = inject_mode
        self.verbose = verbose

        # Early stop
        if early_stop:
            if isinstance(early_stop, EarlyStop):
                self.early_stop = early_stop
            else:
                self.early_stop = EarlyStop(patience=early_stop_patience, epsilon=early_stop_epsilon)
        else:
            self.early_stop = None

    def attack(self, model, graph, feat_norm=None, adj_norm_func=None):
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
        feat_norm : str, optional
            Type of feature normalization, ['arctan', 'tanh']. Default: ``None``.
        adj_norm_func : func of utils.normalize, optional
            Function that normalizes adjacency matrix. Default: ``None``.

        Returns
        -------
        out_graph : cogdl.data.Graph
            Graph attacked.

        """
        time_start = time.time()
        adj = graph.to_scipy_csr()
        target_mask = graph.test_mask
        features = graph.x
        model.to(self.device)
        n_total, n_feat = features.shape
        features = feat_preprocess(features=features, feat_norm=feat_norm, device=self.device)
        adj_tensor = adj_preprocess(adj=adj, adj_norm_func=adj_norm_func, device=self.device)
        pred_origin = model(getGraph(adj_tensor, features, device=self.device))
        labels_origin = torch.argmax(pred_origin, dim=1)
        adj_attack = self.injection(
            adj=adj, n_inject=self.n_inject_max, n_node=n_total, target_mask=target_mask, mode=self.inject_mode
        )

        features_attack = np.zeros([self.n_inject_max, n_feat])
        features_attack = self.update_features(
            model=model,
            adj_attack=adj_attack,
            features_origin=features,
            features_attack=features_attack,
            labels_origin=labels_origin,
            target_mask=target_mask,
            feat_norm=feat_norm,
            adj_norm_func=adj_norm_func,
        )
        out_features = torch.cat((features, features_attack), 0)
        time_end = time.time()
        if self.verbose:
            print("Attack runtime: {:.4f}.".format(time_end - time_start))

        out_graph = getGraph(adj_attack, out_features, graph.y, device=self.device)
        return out_graph

    # flake8: noqa: C901
    def injection(self, adj, n_inject, n_node, target_mask, mode="random-inter"):
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
        mode : str, optional
            Mode of injection. Choose from ``["random", "random-inter", "multi-layer"]``. Default: ``random-inter``.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.

        """

        def get_inject_list(adj, n_inter, inject_id, inject_tmp_list):
            i = 1
            res = []
            while len(res) < n_inter and i < len(inject_tmp_list):
                if adj[inject_id, inject_tmp_list[i]] == 0:
                    res.append(inject_tmp_list[i])
                i += 1

            return res

        def update_active_edges(active_edges, n_inject_edges, threshold):
            for i in active_edges:
                if n_inject_edges[i] >= threshold:
                    active_edges.pop(active_edges.index(i))

            return active_edges

        def inject(target_node_list, n_inject, n_test, mode="random-inter"):
            n_target = len(target_node_list)
            adj = np.zeros((n_inject, n_test + n_inject))

            if mode == "random-inter":
                # target_node_list: a list of target nodes to be attacked
                n_inject_edges = np.zeros(n_inject)
                active_edges = [i for i in range(n_inject)]

                # create edges between injected node and target node
                for i in range(n_target):
                    if not active_edges:
                        break
                    inject_id = np.random.choice(active_edges, 1)
                    n_inject_edges[inject_id] += 1
                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=self.n_edge_max)
                    adj[inject_id, target_node_list[i]] = 1

                # create edges between injected nodes
                for i in range(len(active_edges)):
                    if not active_edges:
                        break
                    inject_tmp_list = sorted(active_edges, key=lambda x: n_inject_edges[x])
                    inject_id = inject_tmp_list[0]
                    n_inter = self.n_edge_max - n_inject_edges[inject_id]
                    inject_list = get_inject_list(adj, n_inter, inject_id, inject_tmp_list)

                    n_inject_edges[inject_list] += 1
                    n_inject_edges[inject_id] += len(inject_list)
                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=self.n_edge_max)
                    if inject_list:
                        adj[inject_id, n_test + np.array(inject_list)] = 1
                        adj[inject_list, n_test + inject_id] = 1

            elif mode == "multi-layer":
                n_inject_edges = np.zeros(n_inject)
                n_inject_layer_1, n_inject_layer_2 = int(n_inject * 0.9), int(n_inject * 0.1)
                n_edge_max_layer_1 = int(self.n_edge_max * 0.9)
                active_edges = [i for i in range(n_inject_layer_1)]

                # create edges between noise node and test node
                for i in range(n_target):
                    if not active_edges:
                        break
                    inject_id = np.random.choice(active_edges, 1)
                    n_inject_edges[inject_id] += 1
                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=n_edge_max_layer_1)
                    adj[inject_id, target_node_list[i]] = 1

                # create edges between injected nodes
                for i in range(len(active_edges)):
                    if not active_edges:
                        break
                    inject_tmp_list = sorted(active_edges, key=lambda x: n_inject_edges[x])
                    inject_id = inject_tmp_list[0]
                    n_inter = n_edge_max_layer_1 - n_inject_edges[inject_id]
                    inject_list = get_inject_list(adj, n_inter, inject_id, inject_tmp_list)

                    n_inject_edges[inject_list] += 1
                    n_inject_edges[inject_id] += len(inject_list)

                    active_edges = update_active_edges(active_edges, n_inject_edges, threshold=n_edge_max_layer_1)

                    if inject_list:
                        adj[inject_id, n_test + np.array(inject_list)] = 1
                        adj[inject_list, n_test + inject_id] = 1

                noise_active_layer2 = [i for i in range(n_inject_layer_2)]
                noise_edge_layer2 = np.zeros(n_inject_layer_2)
                for i in range(n_inject_layer_1):
                    if not noise_active_layer2:
                        break
                    inject_list = np.random.choice(noise_active_layer2, 10)
                    noise_edge_layer2[inject_list] += 1
                    noise_active_layer2 = update_active_edges(
                        noise_active_layer2, noise_edge_layer2, threshold=self.n_edge_max
                    )
                    adj[inject_list + n_inject_layer_1, i + n_test] = 1
                    adj[i, inject_list + n_inject_layer_1 + n_test] = 1
            else:
                print("Mode ERROR: 'mode' should be one of ['random', 'random-inter', 'multi-layer']")

            return adj

        if mode == "random":
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
        else:
            # construct injected adjacency matrix
            test_index = torch.where(target_mask)[0]
            n_test = test_index.shape[0]
            target_node = test_index.cpu().detach().numpy()
            adj_inject = inject(target_node, n_inject, n_test, mode)
            adj_inject = sp.hstack([sp.csr_matrix((n_inject, n_node - n_test)), adj_inject])
            adj_inject = sp.csr_matrix(adj_inject)
            adj_attack = sp.vstack([adj, adj_inject[:, :n_node]])
            adj_attack = sp.hstack([adj_attack, adj_inject.T])
            adj_attack = sp.coo_matrix(adj_attack)

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
        Adversarial feature generation of injected nodes.

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
        adj_norm_func : func of utils.normalize
            Function that normalizes adjacency matrix.

        Returns
        -------
        features_attack : torch.FloatTensor
            Updated features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.

        """

        lr = self.lr
        n_epoch = self.n_epoch
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        n_total = features_origin.shape[0]
        features_attack = feat_preprocess(features=features_attack, feat_norm=feat_norm, device=self.device)
        adj_attacked = adj_preprocess(adj=adj_attack, adj_norm_func=adj_norm_func, device=self.device)
        features_attack.requires_grad_(True)
        optimizer = torch.optim.Adam([features_attack], lr=lr)
        model.eval()
        epoch_bar = tqdm(range(n_epoch))
        for i in epoch_bar:
            features_concat = torch.cat((features_origin, features_attack), dim=0)
            pred = model(getGraph(adj_attacked, features_concat, device=self.device))
            pred_loss = -self.loss(pred[:n_total][target_mask], labels_origin[target_mask]).to(self.device)
            optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                features_attack.clamp_(feat_lim_min, feat_lim_max)

            test_score = eval_acc(pred[:n_total][target_mask], labels_origin[target_mask])

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
