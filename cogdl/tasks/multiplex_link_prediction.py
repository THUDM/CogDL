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


@register_task("multiplex_link_prediction")
class MultiplexLinkPrediction(BaseTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=200)
        parser.add_argument("--eval-type", type=str, default='all', nargs='+')
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(MultiplexLinkPrediction, self).__init__(args)

        dataset = build_dataset(args) if dataset is None else dataset
        data = dataset[0]
        self.data = data
        if hasattr(dataset, "num_features"):
            args.num_features = dataset.num_features
        model = build_model(args) if model is None else model
        self.model = model
        self.eval_type = args.eval_type

    def train(self):
        total_roc_auc, total_f1_score, total_pr_auc = [], [], []
        if hasattr(self.model, "multiplicity"):
            all_embs = self.model.train(self.data.train_data)
        for key in self.data.train_data.keys():
            if self.eval_type == "all" or key in self.eval_type:
                embs = dict()
                if not hasattr(self.model, "multiplicity"):
                    G = nx.Graph()
                    G.add_edges_from(self.data.train_data[key])
                    embeddings = self.model.train(G)

                    for vid, node in enumerate(G.nodes()):
                        embs[node] = embeddings[vid]
                else:
                    embs = all_embs[key]
                roc_auc, f1_score, pr_auc = evaluate(
                    embs, self.data.test_data[key][0], self.data.test_data[key][1]
                )
                total_roc_auc.append(roc_auc)
                total_f1_score.append(f1_score)
                total_pr_auc.append(pr_auc)
        assert len(total_roc_auc) > 0
        roc_auc, f1_score, pr_auc = (
            np.mean(total_roc_auc),
            np.mean(total_f1_score),
            np.mean(total_pr_auc),
        )
        print(
            f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}"
        )
        return dict(ROC_AUC=roc_auc, PR_AUC=pr_auc, F1=f1_score)
