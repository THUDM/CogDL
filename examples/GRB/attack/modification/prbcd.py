import math
import time
from typing import Tuple

import numpy as np
import torch
import torch_sparse
from tqdm.auto import tqdm

from ..base import ModificationAttack, EarlyStop
from cogdl.data import Graph
from cogdl.utils.grb_utils import (
    eval_acc,
    feat_preprocess,
    adj_preprocess,
    getGraph,
    getGRBGraph,
    SPARSEAdjNorm,
    adj_to_tensor,
)


def tanh_margin_loss(logits, labels, normalizer=40):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
    margin = logits[np.arange(logits.size(0)), labels] - logits[np.arange(logits.size(0)), best_non_target_class]
    loss = torch.tanh(-margin / normalizer).mean()
    return loss


class PRBCD(ModificationAttack):
    def __init__(
        self,
        epsilon,  # L_infty budget?
        n_epoch,
        n_node_mod,
        n_edge_mod,
        feat_lim_min,
        feat_lim_max,
        n_epoch_resampling=None,
        allow_isolate=True,
        loss=tanh_margin_loss,
        eval_metric=eval_acc,
        early_stop=None,
        early_stop_patience=1000,
        early_stop_epsilon=1e-5,
        max_final_samples=20,
        eps: float = 1e-7,
        search_space_size: int = 1_000_000,
        make_undirected: bool = True,
        lr_factor: float = 100,
        device="cpu",
        verbose=True,
    ):
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.n_epoch_resampling = int(0.75 * n_epoch) if n_epoch_resampling is None else n_epoch_resampling
        self.n_node_mod = n_node_mod
        self.n_edge_mod = n_edge_mod
        assert allow_isolate, "PRBCD currently only supports `allow_isolate=True`"
        self.allow_isolate = allow_isolate
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.device = device
        self.verbose = verbose

        # Early stop
        # TODO: discuss how to implement
        if early_stop:
            if isinstance(early_stop, EarlyStop):
                self.early_stop = early_stop
            else:
                self.early_stop = EarlyStop(patience=early_stop_patience, epsilon=early_stop_epsilon)
        else:
            self.early_stop = None

        self.make_undirected = make_undirected
        self.max_final_samples = max_final_samples
        self.eps = eps
        self.search_space_size = search_space_size
        self.lr_factor = lr_factor

    def attack(self, model, graph: Graph, feat_norm=None, adj_norm_func=SPARSEAdjNorm):
        time_start = time.time()
        model.to(self.device)
        adj = graph.to_scipy_csr()
        features = graph.x.clone().detach()

        self.n = adj.shape[0]
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        self.lr_factor = self.lr_factor * max(math.log2(self.n_possible_edges / self.search_space_size), 1.0)

        features = feat_preprocess(features=features, feat_norm=feat_norm, device=self.device)
        adj_tensor = adj_preprocess(adj=adj, adj_norm_func=adj_norm_func, device=self.device)
        pred_origin = model(getGraph(adj_tensor, features, device=self.device))
        labels_origin = torch.argmax(pred_origin, dim=1)

        adj_attack, features_attack = self.modification(
            model,
            adj,
            features_origin=features,
            labels_origin=labels_origin,
            index_target=graph.test_nid.cpu(),
            feat_norm=feat_norm,
            adj_norm_func=adj_norm_func,
        )

        time_end = time.time()
        if self.verbose:
            print("Attack runtime: {:.4f}.".format(time_end - time_start))

        return getGraph(adj_attack, features_attack, graph.y, device=self.device)

    def modification(
        self, model, adj, features_origin, labels_origin, index_target, feat_norm=None, adj_norm_func=None
    ):
        r"""

        Description
        -----------
        Attack features and graph structure simultaneously

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.
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
        features_attack = feat_preprocess(features=features_origin, feat_norm=feat_norm, device=self.device)

        index_target = torch.where(index_target)[0]
        index_mod = np.random.choice(index_target, self.n_node_mod)

        assert (adj.todense().T == adj.todense()).all(), "adjacency matrix must be symmetrical"
        adj = adj_to_tensor(adj).to(self.device).coalesce()

        perturbed_edge_indices, perturbed_edge_weight, current_search_space = self.sample_random_block()

        # For early stopping (not explicitly covered by pesudo code)
        best_test_score = float("Inf")
        best_epoch = float("-Inf")

        def evaluate(features, adj):
            pred = model(getGraph(adj_norm_func(adj), features, device=self.device))
            pred_loss = self.loss(pred[index_target], labels_origin[index_target])

            test_score = self.eval_metric(pred[index_target], labels_origin[index_target])
            return pred_loss, test_score

        model.eval()
        epoch_bar = tqdm(range(self.n_epoch), disable=not self.verbose)
        for epoch in epoch_bar:
            features_attack.requires_grad_(True)
            perturbed_edge_weight.requires_grad_(True)

            modified_adj = self.get_modified_adj(adj, perturbed_edge_indices, perturbed_edge_weight)
            pred_loss, test_score = evaluate(features_attack, modified_adj)

            if self.early_stop:
                if test_score < best_test_score:
                    best_epoch = epoch
                    best_test_score = test_score
                    best_perturbed_edge_indices = perturbed_edge_indices.cpu().detach()
                    best_perturbed_edge_weight = perturbed_edge_weight.cpu().detach()
                    best_current_search_space = current_search_space.cpu().detach()
                    best_features_attack = features_attack.cpu().detach()
                self.early_stop(test_score)
                if self.early_stop.stop:
                    self.early_stop = EarlyStop()
                    if self.verbose:
                        print("Attack early stopped. Surrogate test score: {:.4f}".format(test_score))
                        print(f"Loading search space of epoch {best_epoch} for fine tuning\n")
                    break

            if self.verbose:
                epoch_bar.set_description(
                    "Epoch {}, Loss: {:.4f}, Surrogate test score: {:.4f}".format(epoch, pred_loss, test_score)
                )

            model.zero_grad()
            pred_loss.backward()

            with torch.no_grad():
                # Feature update (as in modification.pgd)
                grad = features_attack.grad.data
                features_attack = features_attack.detach()
                features_attack[index_mod] = features_attack.clone()[index_mod] + self.epsilon * grad.sign()[index_mod]
                features_attack = torch.clamp(features_attack, self.feat_lim_min, self.feat_lim_max)

                # Update edge weights
                grad = perturbed_edge_weight.grad.data
                perturbed_edge_weight = self.update_edge_weights(perturbed_edge_weight, epoch, grad)
                perturbed_edge_weight = self.project(perturbed_edge_weight)

                # Resample random block for `self.n_epoch_resampling` epochs
                if epoch < self.n_epoch_resampling - 1:
                    perturbed_edge_indices, perturbed_edge_weight, current_search_space = self.resample_random_block(
                        perturbed_edge_indices, perturbed_edge_weight, current_search_space
                    )
                # Take the best random block including edge weights for fine tuning
                elif self.early_stop and epoch == self.n_epoch_resampling - 1:
                    if self.verbose:
                        print(f"Fine tune block of epoch {best_epoch} (test_score={best_test_score})\n")
                    perturbed_edge_indices = best_perturbed_edge_indices.to(self.device)
                    perturbed_edge_weight = best_perturbed_edge_weight.to(self.device)
                    current_search_space = best_current_search_space.to(self.device)
                    features_attack = best_features_attack.to(self.device)

        if self.verbose:
            print("Surrogate test score: {:.4f}".format(best_test_score))
            print(f"Loading search space of epoch {best_epoch}\n")

        if self.early_stop:
            perturbed_edge_indices = best_perturbed_edge_indices.to(self.device)
            perturbed_edge_weight = best_perturbed_edge_weight.to(self.device)
            current_search_space = best_current_search_space.to(self.device)
            features_attack = best_features_attack.to(self.device)

        modified_adj = self.sample_final_edges(
            evaluate, adj, features_attack, perturbed_edge_indices, perturbed_edge_weight
        )

        return modified_adj, features_attack

    def sample_random_block(self):
        for _ in range(self.max_final_samples):
            current_search_space = torch.randint(self.n_possible_edges, (self.search_space_size,), device=self.device)
            current_search_space = torch.unique(current_search_space, sorted=True)
            if self.make_undirected:
                perturbed_edge_indices = PRBCD.linear_to_triu_idx(self.n, current_search_space)
            else:
                perturbed_edge_indices = PRBCD.linear_to_full_idx(self.n, current_search_space)
                is_not_self_loop = perturbed_edge_indices[0] != perturbed_edge_indices[1]
                current_search_space = current_search_space[is_not_self_loop]
                perturbed_edge_indices = perturbed_edge_indices[:, is_not_self_loop]

            perturbed_edge_weight = torch.full_like(
                current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if current_search_space.size(0) >= self.n_edge_mod:
                return perturbed_edge_indices, perturbed_edge_weight, current_search_space
        raise RuntimeError("Sampling random block was not successfull. Please decrease `n_edge_mod`.")

    def get_modified_adj(self, adj, perturbed_edge_indices, perturbed_edge_weight):
        if self.make_undirected:
            perturbed_edge_indices, perturbed_edge_weight = PRBCD.to_symmetric(
                perturbed_edge_indices, perturbed_edge_weight, self.n
            )
        edge_index = torch.cat((adj.indices().to(self.device), perturbed_edge_indices), dim=-1)
        edge_weight = torch.cat((adj.values().to(self.device), perturbed_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op="sum")

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]

        return torch.sparse.FloatTensor(edge_index, edge_weight, (self.n, self.n)).coalesce()

    def update_edge_weights(
        self, perturbed_edge_weight: torch.Tensor, epoch: int, gradient: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the edge weights and adaptively, heuristically refined the learning rate such that (1) it is
        independent of the number of perturbations (assuming an undirected adjacency matrix) and (2) to decay learning
        rate during fine-tuning (i.e. fixed search space).

        Parameters
        ----------
        perturbed_edge_weight : torch.Tensor
            Perturbed edge weights to be updated.
        epoch : int
            Number of epochs until fine tuning.
        gradient : torch.Tensor
            The current gradient.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated edge indices and weights.
        """
        # For the learning rate to be orthogonal to the block and graph size
        lr_factor = self.n_edge_mod / self.n / 2 * self.lr_factor
        # Once we perform PGD with fixed edges we decay the learning rate as originally proposed
        lr = lr_factor / np.sqrt(max(0, epoch - self.n_epoch_resampling) + 1)

        perturbed_edge_weight += lr * gradient

        # We require for technical reasons that all edges in the block have at least a small positive value
        perturbed_edge_weight.data[perturbed_edge_weight < self.eps] = self.eps

        return perturbed_edge_weight

    def project(self, values: torch.Tensor, eps: float = 0, inplace: bool = False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > self.n_edge_mod:
            left = (values - 1).min()
            right = values.max()
            miu = PRBCD.bisection(values, left, right, self.n_edge_mod)
            values.data.copy_(torch.clamp(values - miu, min=eps, max=1 - eps))
        else:
            values.data.copy_(torch.clamp(values, min=eps, max=1 - eps))
        return values

    def resample_random_block(self, perturbed_edge_indices, perturbed_edge_weight, current_search_space):
        sorted_idx = torch.argsort(perturbed_edge_weight)
        idx_keep = (perturbed_edge_weight <= self.eps).sum().long()

        # Keep at most half of the block (i.e. resample low weights)
        if idx_keep < sorted_idx.size(0) // 2:
            idx_keep = sorted_idx.size(0) // 2

        sorted_idx = sorted_idx[idx_keep:]
        current_search_space = current_search_space[sorted_idx]
        perturbed_edge_indices = perturbed_edge_indices[:, sorted_idx]
        perturbed_edge_weight = perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = self.search_space_size - current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            current_search_space, unique_idx = torch.unique(
                torch.cat((current_search_space, lin_index)), sorted=True, return_inverse=True
            )

            if self.make_undirected:
                perturbed_edge_indices = PRBCD.linear_to_triu_idx(self.n, current_search_space)
            else:
                perturbed_edge_indices = PRBCD.linear_to_full_idx(self.n, current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = perturbed_edge_weight.clone()
            perturbed_edge_weight = torch.full_like(current_search_space, self.eps, dtype=torch.float32)
            perturbed_edge_weight[unique_idx[: perturbed_edge_weight_old.size(0)]] = perturbed_edge_weight_old

            if not self.make_undirected:
                is_not_self_loop = perturbed_edge_indices[0] != perturbed_edge_indices[1]
                current_search_space = current_search_space[is_not_self_loop]
                perturbed_edge_indices = perturbed_edge_indices[:, is_not_self_loop]
                perturbed_edge_weight = perturbed_edge_weight[is_not_self_loop]

            if current_search_space.size(0) > self.n_edge_mod:
                return perturbed_edge_indices, perturbed_edge_weight, current_search_space
        raise RuntimeError("Sampling random block was not successfull. Please decrease `n_perturbations`.")

    @torch.no_grad()
    def sample_final_edges(
        self, evaluate, adj, features, perturbed_edge_indices, perturbed_edge_weight
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        best_test_score = float("Inf")
        perturbed_edge_weight = perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        for i in range(self.max_final_samples):
            if best_test_score == float("Inf"):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, self.n_edge_mod).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > self.n_edge_mod:
                # Budget is violated -> ignore the sample
                continue

            modified_adj = self.get_modified_adj(adj, perturbed_edge_indices, sampled_edges)
            _, test_score = evaluate(features, modified_adj)

            # Save best sample
            if best_test_score > test_score:
                best_test_score = test_score
                best_edges = sampled_edges.clone().cpu()

        # Recover best sample
        perturbed_edge_weight = best_edges.to(self.device)

        edge_mask = perturbed_edge_weight == 1
        modified_adj = self.get_modified_adj(
            adj, perturbed_edge_indices[:, edge_mask], perturbed_edge_weight[edge_mask]
        )

        allowed_perturbations = 2 * self.n_edge_mod if self.make_undirected else self.n_edge_mod
        edges_after_attack = len(modified_adj.values())
        clean_edges = len(adj.values())
        assert (
            edges_after_attack >= clean_edges - allowed_perturbations
            and edges_after_attack <= clean_edges + allowed_perturbations
        ), f"{edges_after_attack} out of range with {clean_edges} clean edges and {self.n_edge_mod} pertutbations"
        return modified_adj

    @staticmethod
    def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = (n - 2 - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)).long()
        col_idx = lin_idx + row_idx + 1 - n * (n - 1) // 2 + (n - row_idx) * ((n - row_idx) - 1) // 2
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = lin_idx // n
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def to_symmetric(
        edge_index: torch.Tensor, edge_weight: torch.Tensor, n: int, op="mean"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        symmetric_edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)

        symmetric_edge_weight = edge_weight.repeat(2)

        symmetric_edge_index, symmetric_edge_weight = torch_sparse.coalesce(
            symmetric_edge_index, symmetric_edge_weight, m=n, n=n, op=op
        )
        return symmetric_edge_index, symmetric_edge_weight

    @staticmethod
    def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
        def func(x):
            return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

        miu = a
        for i in range(int(iter_max)):
            miu = (a + b) / 2
            # Check if middle point is root
            if func(miu) == 0.0:
                break
            # Decide the side to repeat the steps
            if func(miu) * func(a) < 0:
                b = miu
            else:
                a = miu
            if (b - a) <= epsilon:
                break

        return miu
