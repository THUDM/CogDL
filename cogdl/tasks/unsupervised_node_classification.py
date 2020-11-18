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
from cogdl.data import Dataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model, register_model

from . import BaseTask, register_task

try:
    from torch_geometric.data import InMemoryDataset
except ImportError:
    pyg = False
else:
    pyg = True

warnings.filterwarnings("ignore")


@register_task("unsupervised_node_classification")
class UnsupervisedNodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-shuffle", type=int, default=5)
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(UnsupervisedNodeClassification, self).__init__(args)
        dataset = build_dataset(args) if dataset is None else dataset

        self.data = dataset[0]
        if pyg and issubclass(dataset.__class__.__bases__[0], InMemoryDataset):
            self.num_nodes = self.data.y.shape[0]
            self.num_classes = dataset.num_classes
            self.label_matrix = np.zeros((self.num_nodes, self.num_classes), dtype=int)
            self.label_matrix[range(self.num_nodes), self.data.y] = 1
            self.data.edge_attr = self.data.edge_attr.t()
        else:
            self.label_matrix = self.data.y
            self.num_nodes, self.num_classes = self.data.y.shape

        args.num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else 0
        args.num_features = dataset.num_features if hasattr(dataset, 'num_features') else 0
        self.model = build_model(args) if model is None else model

        self.model_name = args.model
        self.dataset_name = args.dataset
        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle
        self.save_dir = args.save_dir
        self.enhance = args.enhance
        self.args = args
        self.is_weighted = self.data.edge_attr is not None

    def enhance_emb(self, G, embs):
        A = sp.csr_matrix(nx.adjacency_matrix(G))
        if self.args.enhance == "prone":
            self.args.model = 'prone'
            self.args.step, self.args.theta, self.args.mu = 5, 0.5, 0.2
            model = build_model(self.args)
            embs = model._chebyshev_gaussian(A, embs)
        elif self.args.enhance == "prone++":
            self.args.model = "prone++"
            self.args.filter_types = ["heat", "ppr", "gaussian", "sc"]
            if not hasattr(self.args, "max_evals"):
                self.args.max_evals = 100
            if not hasattr(self.args, "num_workers"):
                self.args.num_workers = 10
            if not hasattr(self.args, "no_svd"):
                self.args.no_svd = False
            self.args.loss = "infomax"
            self.args.no_search = False
            model = build_model(self.args)
            embs = model(embs, A)
        else:
            raise ValueError("only supports 'prone' and 'prone++'")
        return embs

    def save_emb(self, embs):
        name = os.path.join(self.save_dir, self.model_name + '_emb.npy')
        np.save(name, embs)

    def train(self):
        if 'gcc' in self.model_name:
            features_matrix = self.model.train(self.data)
        elif 'dgi' in self.model_name or "graphsage" in self.model_name:
            acc = self.model.train(self.data)
            return dict(Acc=acc)
        elif 'mvgrl' in self.model_name:
            acc = self.model.train(self.data, self.dataset_name)
            return dict(Acc=acc)
        else:
            G = nx.Graph()
            if self.is_weighted:
                edges, weight = (
                    self.data.edge_index.t().tolist(),
                    self.data.edge_attr.tolist(),
                )
                G.add_weighted_edges_from(
                    [(edges[i][0], edges[i][1], weight[0][i]) for i in range(len(edges))]
                )
            else:
                G.add_edges_from(self.data.edge_index.t().tolist())
            embeddings = self.model.train(G)
            if self.enhance is not None:
                embeddings = self.enhance_emb(G, embeddings)
            # Map node2id
            features_matrix = np.zeros((self.num_nodes, self.hidden_size))
            for vid, node in enumerate(G.nodes()):
                features_matrix[node] = embeddings[vid]

        self.save_emb(features_matrix)

        # label nor multi-label
        label_matrix = sp.csr_matrix(self.label_matrix)

        return self._evaluate(features_matrix, label_matrix, self.num_shuffle)

    def _evaluate(self, features_matrix, label_matrix, num_shuffle):
        # features_matrix, node2id = utils.load_embeddings(args.emb)
        # label_matrix = utils.load_labels(args.label, node2id, divi_str=" ")

        # shuffle, to create train/test groups
        shuffles = []
        for _ in range(num_shuffle):
            shuffles.append(skshuffle(features_matrix, label_matrix))

        # score each train/test group
        all_results = defaultdict(list)
        # training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]

        for train_percent in training_percents:
            for shuf in shuffles:
                X, y = shuf

                training_size = int(train_percent * self.num_nodes)

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
            # print("micro", result)

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
