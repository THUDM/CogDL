import argparse
import copy
import os

import numpy as np
import torch
from cogdl.data import Graph, DataLoader
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from . import BaseTask, register_task
from .graph_classification import node_degree_as_feature


@register_task("unsupervised_graph_classification")
class UnsupervisedGraphClassification(BaseTask):
    r"""Unsupervised graph classification"""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--num-shuffle", type=int, default=10)
        parser.add_argument("--degree-feature", dest="degree_feature", action="store_true")
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(UnsupervisedGraphClassification, self).__init__(args)

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]

        dataset = build_dataset(args) if dataset is None else dataset
        if "gcc" in args.model:
            self.label = dataset.graph_labels[:, 0]
            self.data = dataset.graph_lists
        else:
            self.label = np.array([data.y for data in dataset])
            self.data = [
                Graph(x=data.x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr).apply(
                    lambda x: x.to(self.device)
                )
                for data in dataset
            ]
        args.num_features = dataset.num_features
        args.num_classes = args.hidden_size
        args.use_unsup = True

        if args.degree_feature:
            self.data = node_degree_as_feature(self.data)
            args.num_features = self.data[0].num_features

        self.num_graphs = len(self.data)
        self.num_classes = dataset.num_classes
        # self.label_matrix = np.zeros((self.num_graphs, self.num_classes))
        # self.label_matrix[range(self.num_graphs), np.array([data.y for data in self.data], dtype=int)] = 1

        self.model = build_model(args) if model is None else model
        self.model = self.model.to(self.device)
        self.model_name = args.model
        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle
        self.save_dir = args.save_dir
        self.epoch = args.epoch
        self.use_nn = args.model in ("infograph",)

        if self.use_nn:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.data_loader = DataLoader(self.data, batch_size=args.batch_size, shuffle=True)

    def train(self):
        if self.use_nn:
            best_model = None
            best_loss = 10000
            epoch_iter = tqdm(range(self.epoch))
            for epoch in epoch_iter:
                loss_n = []
                for batch in self.data_loader:
                    batch = batch.to(self.device)
                    loss = self.model.graph_classification_loss(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_n.append(loss.item())
                loss_n = np.mean(loss_n)
                epoch_iter.set_description(f"Epoch: {epoch:03d}, TrainLoss: {np.mean(loss_n)} ")
                if loss_n < best_loss:
                    best_loss = loss_n
                    best_model = copy.deepcopy(self.model)
            self.model = best_model
            with torch.no_grad():
                self.model.eval()
                prediction = []
                label = []
                for batch in self.data_loader:
                    batch = batch.to(self.device)
                    predict = self.model(batch)
                    prediction.extend(predict.cpu().numpy())
                    label.extend(batch.y.cpu().numpy())
                prediction = np.array(prediction).reshape(len(label), -1)
                label = np.array(label).reshape(-1)
        elif "gcc" in self.model_name:
            prediction = self.model.train(self.data)
            label = self.label
        else:
            prediction = self.model(self.data)
            label = self.label

        if prediction is not None:
            # self.save_emb(prediction)
            return self._evaluate(prediction, label)

    def save_emb(self, embs):
        name = os.path.join(self.save_dir, self.model_name + "_emb.npy")
        np.save(name, embs)

    def _evaluate(self, embeddings, labels):
        result = []
        kf = KFold(n_splits=10)
        kf.get_n_splits(X=embeddings, y=labels)
        for train_index, test_index in kf.split(embeddings):
            x_train = embeddings[train_index]
            x_test = embeddings[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]
            params = {"C": [1e-2, 1e-1, 1]}
            svc = SVC()
            clf = GridSearchCV(svc, params)
            clf.fit(x_train, y_train)

            preds = clf.predict(x_test)
            f1 = f1_score(y_test, preds, average="micro")
            result.append(f1)
        test_f1 = np.mean(result)
        test_std = np.std(result)

        print("Test Acc: ", test_f1)
        return dict(Acc=test_f1, Std=test_std)
