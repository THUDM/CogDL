import argparse
import copy

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from cogdl.data import DataLoader
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import add_remaining_self_loops

from . import BaseTask, register_task


def node_degree_as_feature(data):
    r"""
    Set each node feature as one-hot encoding of degree
    :param data: a list of class Data
    :return: a list of class Data
    """
    max_degree = 0
    degrees = []
    for graph in data:
        row, col = graph.edge_index
        edge_weight = torch.ones((row.shape[0],), device=row.device)
        fill_value = 1
        num_nodes = graph.num_nodes
        (row, col), edge_weight = add_remaining_self_loops((row, col), edge_weight, fill_value, num_nodes)
        deg = torch.zeros(num_nodes).to(row.device).scatter_add_(0, row, edge_weight).long()
        degrees.append(deg.cpu() - 1)
        max_degree = max(torch.max(deg), max_degree)
    max_degree = int(max_degree)
    for i in range(len(data)):
        one_hot = torch.zeros(data[i].num_nodes, max_degree).scatter_(1, degrees[i].unsqueeze(1), 1)
        data[i].x = one_hot.to(data[i].y.device)
    return data


def uniform_node_feature(data):
    r"""Set each node feature to the same"""
    feat_dim = 2
    init_feat = torch.rand(1, feat_dim)
    for i in range(len(data)):
        data[i].x = init_feat.repeat(1, data[i].num_nodes)
    return data


@register_task("graph_classification")
class GraphClassification(BaseTask):
    r"""Superiviced graph classification task."""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--degree-feature", dest="degree_feature", action="store_true")
        parser.add_argument("--gamma", type=float, default=0.5)
        parser.add_argument("--uniform-feature", action="store_true")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--kfold", dest="kfold", action="store_true")
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(GraphClassification, self).__init__(args)
        dataset = build_dataset(args) if dataset is None else dataset

        args.max_graph_size = max([ds.num_nodes for ds in dataset])
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.use_unsup = False

        self.args = args
        self.kfold = args.kfold
        self.folds = 10

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]

        if args.dataset.startswith("ogbg"):
            self.data = dataset.data
            self.train_loader, self.val_loader, self.test_loader = dataset.get_loader(args)
            model = build_model(args) if model is None else model
        else:
            self.data = dataset
            if self.data[0].x is None:
                self.data = node_degree_as_feature(dataset)
                args.num_features = self.data.num_features
            model = build_model(args) if model is None else model
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = model.split_dataset(self.data, args)
            self.train_loader = DataLoader(**self.train_dataset)
            self.val_loader = DataLoader(**self.val_dataset)
            self.test_loader = DataLoader(**self.test_dataset)

        self.model = model.to(self.device)

        self.set_loss_fn(dataset)
        self.set_evaluator(dataset)

        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50, gamma=0.5)

    def train(self):
        if self.kfold:
            return self._kfold_train()
        else:
            return self._train()

    def _train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_model = None
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf

        for epoch in epoch_iter:
            self.scheduler.step()
            self._train_step()
            train_acc, train_loss = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="valid")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, TrainLoss:{train_loss: .4f}, ValLoss: {val_loss: .4f}"
            )
            if val_loss < min_loss or val_acc > max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                patience += 1
                if patience == self.patience:
                    self.model = best_model
                    epoch_iter.close()
                    break
        self.model = best_model
        print(len(self.test_loader), len(self.train_loader))
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="valid")
        print(f"Test accuracy = {test_acc}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _train_step(self):
        self.model.train()
        loss_n = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.graph_classification_loss(batch)
            loss_n += loss.item()
            loss.backward()
            self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        if split == "train":
            loader = self.train_loader
        elif split == "test":
            loader = self.test_loader
        elif split == "valid":
            loader = self.val_loader
        else:
            raise ValueError
        loss_n = []
        pred = []
        y = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                prediction = self.model(batch)
                loss = self.loss_fn(prediction, batch.y)
                loss_n.append(loss.item())
                y.append(batch.y)
                pred.extend(prediction)
        y = torch.cat(y).to(self.device)

        pred = torch.stack(pred, dim=0)
        metric = self.evaluator(pred, y)
        return metric, sum(loss_n) / len(loss_n)

    def _kfold_train(self):
        y = [x.y for x in self.data]
        kf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.args.seed)
        acc = []
        for train_index, test_index in kf.split(self.data, y=y):
            model = build_model(self.args)
            self.model = model.to(self.device)
            self.model.set_loss_fn(self.loss_fn)

            droplast = self.args.model == "diffpool"
            self.train_loader = DataLoader(
                [self.data[i] for i in train_index],
                batch_size=self.args.batch_size,
                drop_last=droplast,
            )
            self.test_loader = DataLoader(
                [self.data[i] for i in test_index],
                batch_size=self.args.batch_size,
                drop_last=droplast,
            )
            self.val_loader = DataLoader(
                [self.data[i] for i in test_index],
                batch_size=self.args.batch_size,
                drop_last=droplast,
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50, gamma=0.5)

            res = self._train()
            acc.append(res["Acc"])
        return dict(Acc=np.mean(acc), Std=np.std(acc))
