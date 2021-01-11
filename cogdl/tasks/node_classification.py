import argparse
import copy
from typing import Optional
import scipy.sparse as sp

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.models.supervised_model import SupervisedHomogeneousNodeClassificationModel
from cogdl.trainers.supervised_trainer import (
    SupervisedHomogeneousNodeClassificationTrainer,
)
from cogdl.trainers.sampled_trainer import SAINTTrainer

from . import BaseTask, register_task


def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx


def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_l1_normalize(X):
    norm = 1e-6 + X.sum(dim=1, keepdim=True)
    return X / norm


def preprocess_data(data, normalize_feature=True, missing_rate=0):
    data.train_mask = data.train_mask.type(torch.bool)
    data.val_mask = data.val_mask.type(torch.bool)
    # expand test_mask to all rest nodes
    data.test_mask = ~(data.train_mask + data.val_mask)
    # get adjacency matrix
    n = len(data.x)
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    adj = normalize_adj_row(adj)
    data.adj = to_torch_sparse(adj).to_dense()
    if normalize_feature:
        data.x = row_l1_normalize(data.x)
    erasing_pool = torch.arange(n)[~data.train_mask]
    size = int(len(erasing_pool) * (missing_rate / 100))
    idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
    if missing_rate > 0:
        data.x[idx_erased] = 0
    return data


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--missing-rate", type=int, default=0, help="missing rate, from 0 to 100")
        # fmt: on

    def __init__(
        self,
        args,
        dataset=None,
        model: Optional[SupervisedHomogeneousNodeClassificationModel] = None,
    ):
        super(NodeClassification, self).__init__(args)

        self.args = args
        self.model_name = args.model

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        dataset = build_dataset(args) if dataset is None else dataset

        if args.missing_rate > 0:
            dataset.data = preprocess_data(dataset.data, normalize_feature=True, missing_rate=args.missing_rate)

        self.dataset = dataset
        self.data = dataset[0]
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.num_nodes = dataset.data.x.shape[0]

        self.model: SupervisedHomogeneousNodeClassificationModel = build_model(args) if model is None else model
        self.model.set_device(self.device)

        self.trainer: Optional[SupervisedHomogeneousNodeClassificationTrainer] = (
            self.model.get_trainer(NodeClassification, self.args)(self.args)
            if self.model.get_trainer(NodeClassification, self.args)
            else None
        )

        if not self.trainer:
            self.optimizer = (
                torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                if not hasattr(self.model, "get_optimizer")
                else self.model.get_optimizer(args)
            )
            self.data.apply(lambda x: x.to(self.device))
            self.model: SupervisedHomogeneousNodeClassificationModel = self.model.to(self.device)
            self.patience = args.patience
            self.max_epoch = args.max_epoch

    def train(self):
        if self.trainer:
            if isinstance(self.trainer, SAINTTrainer):
                self.model = self.trainer.fit(self.model, self.dataset)
                self.data.apply(lambda x: x.to(self.device))
            else:
                result = self.trainer.fit(self.model, self.dataset)
                if issubclass(type(result), torch.nn.Module):
                    self.model = result
                else:
                    return result
        else:
            epoch_iter = tqdm(range(self.max_epoch))
            patience = 0
            best_score = 0
            best_loss = np.inf
            max_score = 0
            min_loss = np.inf
            best_model = copy.deepcopy(self.model)
            for epoch in epoch_iter:
                self._train_step()
                train_acc, _ = self._test_step(split="train")
                val_acc, val_loss = self._test_step(split="val")
                epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_loss <= best_loss:  # and val_acc >= best_score:
                        best_loss = val_loss
                        best_score = val_acc
                        best_model = copy.deepcopy(self.model)
                    min_loss = np.min((min_loss, val_loss))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
            print(f"Valid accurracy = {best_score}")
            self.model = best_model
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_acc}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.node_classification_loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val", logits=None):
        self.model.eval()
        logits = logits if logits else self.model.predict(self.data)
        logits = F.log_softmax(logits, dim=-1)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        loss = F.nll_loss(logits[mask], self.data.y[mask]).item()

        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss
