import copy
import os
from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.data import Data, DataLoader
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from . import BaseTask, register_task
from .graph_classification import node_degree_as_feature


@register_task("unsupervised_graph_classification")
class UnsupervisedGraphClassification(BaseTask):
    r"""Unsupervised graph classification"""
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--num-shuffle", type=int, default=10)
        parser.add_argument("--degree-feature", dest="degree_feature", action="store_true")
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(UnsupervisedGraphClassification, self).__init__(args)
        self.device = args.device_id[0] if not args.cpu else 'cpu'

        dataset = build_dataset(args) if dataset is None else dataset
        if 'gcc' in args.model:
            self.label = dataset.graph_labels[:, 0]
            self.data = dataset.graph_lists
        else:
            self.label = np.array([data.y for data in dataset])
            self.data = [
                Data(x=data.x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr,
                    pos=data.pos).apply(lambda x:x.to(self.device))
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
        self.use_nn = args.model in ('infograph', )

        if self.use_nn:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
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
                    predict, loss = self.model(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_n.append(loss.item())
                loss_n = np.mean(loss_n)
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, TrainLoss: {np.mean(loss_n)} "
                )
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
                    predict, _ = self.model(batch)
                    prediction.extend(predict.cpu().numpy())
                    label.extend(batch.y.cpu().numpy())
                prediction = np.array(prediction).reshape(len(label), -1)
                label = np.array(label).reshape(-1)
        elif 'gcc' in self.model_name:
            prediction = self.model.train(self.data)
            label = self.label
        else:
            prediction, loss = self.model(self.data)
            label = self.label

        if prediction is not None:
            # self.save_emb(prediction)
            return self._evaluate(prediction, label)

    def save_emb(self, embs):
        name = os.path.join(self.save_dir, self.model_name + '_emb.npy')
        np.save(name, embs)

    def _evaluate(self, embeddings, labels):
        shuffles = []
        for _ in range(self.num_shuffle):
            shuffles.append(skshuffle(embeddings, labels))
        all_results = defaultdict(list)
        training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]

        for training_percent in training_percents:
            for shuf in shuffles:
                training_size = int(training_percent * self.num_graphs)
                X, y = shuf
                X_train = X[:training_size, :]
                y_train = y[:training_size]

                X_test = X[training_size:, :]
                y_test = y[training_size:]
                # clf = SVC()
                # clf.fit(X_train, y_train)

                params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
                svc = SVC()
                clf = GridSearchCV(svc, params)
                clf.fit(X_train, y_train)

                preds = clf.predict(X_test)
                accuracy = f1_score(y_test, preds, average="micro")
                all_results[training_percent].append(accuracy)

        result = dict(
            (
                f"Accuracy {train_percent}",
                sum(all_results[train_percent]) / len(all_results[train_percent]),
            )
            for train_percent in sorted(all_results.keys())
        )
        print("Test Acc: ", list(result.values())[-1])
        return result
