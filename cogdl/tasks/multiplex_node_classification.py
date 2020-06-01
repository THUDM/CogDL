import copy
import os
import random
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from cogdl import options
from cogdl.data import Dataset, InMemoryDataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model, register_model

from . import BaseTask, register_task

warnings.filterwarnings("ignore")


@register_task("multiplex_node_classification")
class MultiplexNodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-shuffle", type=int, default=5)
        # fmt: on

    def __init__(self, args):
        super(MultiplexNodeClassification, self).__init__(args)
        dataset = build_dataset(args)
        self.data = dataset[0]
        self.label_matrix = self.data.y
        self.num_nodes, self.num_classes = self.data.y.shape

        self.model = build_model(args)
        self.model_name = args.model
        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle
        self.save_dir = args.save_dir
        self.enhance = args.enhance
        self.args = args
        self.is_weighted = self.data.edge_attr is not None
    

    def train(self):
        G = nx.DiGraph()
        if self.is_weighted:
            edges, weight = (
                self.data.edge_index.t().tolist(),
                self.data.edge_attr.tolist(),
            )
            G.add_weighted_edges_from(
                [(edges[i][0], edges[i][1], weight[i]) for i in range(len(edges))]
            )
        else:
            G.add_edges_from(self.data.edge_index.t().tolist())
        embeddings = self.model.train(G, self.data.pos.tolist())

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]
            
        # Select nodes which have label as training data
        select_index = [i for i in range(self.num_nodes) if sum(self.label_matrix[i]) !=0]
        features_matrix, label_matrix = features_matrix[select_index], self.label_matrix[select_index]
        label_matrix = sp.csr_matrix(label_matrix)
        return self._evaluate(features_matrix, label_matrix, self.num_shuffle)

    def _evaluate(self, features_matrix, label_matrix, num_shuffle):
        shuffles = []
        for _ in range(num_shuffle):
            shuffles.append(skshuffle(features_matrix, label_matrix))
        all_results = defaultdict(list)
        training_percents = [0.3, 0.5, 0.7, 0.9]

        for train_percent in training_percents:
            for shuf in shuffles:
                X, y = shuf

                training_size = int(train_percent * len(X))

                X_train = X[:training_size, :]
                y_train = y[:training_size, :]

                X_test = X[training_size:, :]
                y_test = y[training_size:, :]

                clf = TopKRanker(LogisticRegression())
                clf.fit(X_train, y_train)

                # find out how many labels should be predicted
                top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
                preds = clf.predict(X_test, top_k_list)
                result = f1_score(y_test, preds, average="micro")
                all_results[train_percent].append(result)

        return dict(
            (
                f"Micro-F1 {train_percent}",
                sum(all_results[train_percent]) / len(all_results[train_percent]),
            )
            for train_percent in sorted(all_results.keys())
        )


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels
