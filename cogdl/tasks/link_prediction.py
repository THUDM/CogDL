import random
from collections import defaultdict

import copy
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.keyedvectors import Vocab
from six import iteritems
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task


def divide_data(input_list, division_rate):
    local_division = len(input_list) * np.cumsum(np.array(division_rate))
    random.shuffle(input_list)
    return [
        input_list[
            int(round(local_division[i - 1]))
            if i > 0
            else 0 : int(round(local_division[i]))
        ]
        for i in range(len(local_division))
    ]


def randomly_choose_false_edges(nodes, true_edges, num):
    true_edges_set = set(true_edges)
    tmp_list = list()
    all_flag = False
    for _ in range(num):
        trial = 0
        while True:
            x = nodes[random.randint(0, len(nodes) - 1)]
            y = nodes[random.randint(0, len(nodes) - 1)]
            trial += 1
            if trial >= 1000:
                all_flag = True
                break
            if x != y and (x, y) not in true_edges_set and (y, x) not in true_edges_set:
                tmp_list.append((x, y))
                break
        if all_flag:
            break
    return tmp_list


def gen_node_pairs(train_data, test_data, negative_ratio=5):
    G = nx.Graph()
    G.add_edges_from(train_data)

    training_nodes = set(list(G.nodes()))
    test_true_data = []
    for u, v in test_data:
        if u in training_nodes and v in training_nodes:
            test_true_data.append((u, v))
    test_false_data = randomly_choose_false_edges(
        list(training_nodes), train_data, len(test_data) * negative_ratio
    )
    return (test_true_data, test_false_data)


def get_score(embs, node1, node2):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def evaluate(embs, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        true_list.append(1)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    for edge in false_edges:
        true_list.append(0)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


@register_task("link_prediction")
class LinkPrediction(BaseTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--negative-ratio", type=int, default=5)
        # fmt: on

    def __init__(self, args):
        super(LinkPrediction, self).__init__(args)

        dataset = build_dataset(args)
        data = dataset[0]
        self.data = data
        if hasattr(dataset, "num_features"):
            args.num_features = dataset.num_features
        model = build_model(args)
        self.model = model
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        edge_list = self.data.edge_index.numpy()
        edge_list = list(zip(edge_list[0], edge_list[1]))
        edge_set = set()
        for edge in edge_list:
            if (edge[0], edge[1]) not in edge_set and (edge[1], edge[0]) not in edge_set:
                edge_set.add(edge)
        edge_list = list(edge_set)
        self.train_data, self.test_data = divide_data(
            edge_list, [0.90, 0.10]
        )

        self.test_data = gen_node_pairs(
            self.train_data, self.test_data, args.negative_ratio
        )

    def train(self):
        G = nx.Graph()
        G.add_edges_from(self.train_data)
        embeddings = self.model.train(G)

        embs = dict()
        for vid, node in enumerate(G.nodes()):
            embs[node] = embeddings[vid]

        roc_auc, f1_score, pr_auc = evaluate(embs, self.test_data[0], self.test_data[1])
        print(
            f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}"
        )
        return dict(ROC_AUC=roc_auc, PR_AUC=pr_auc, F1=f1_score)
