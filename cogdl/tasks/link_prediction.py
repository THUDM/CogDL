import random

import copy
import networkx as nx
import numpy as np
import torch
from torch import mode
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


def select_task(model_name=None, model=None):
    assert model_name is not None or model is not None
    if model_name is not None:
        return model_name in ["rgcn", "compgcn"]
    else:
        from cogdl.models.nn import rgcn, compgcn
        return type(model) in [rgcn.LinkPredictRGCN, compgcn.LinkPredictCompGCN]


class HomoLinkPrediction(nn.Module):
    def __init__(self, args, dataset=None, model=None):
        super(HomoLinkPrediction, self).__init__()
        dataset = build_dataset(args) if dataset is None else dataset
        data = dataset[0]
        self.data = data
        if hasattr(dataset, "num_features"):
            args.num_features = dataset.num_features
        model = build_model(args) if model is None else model
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


class KGLinkPrediction(nn.Module):
    def __init__(self, args, dataset=None, model=None):
        super(KGLinkPrediction, self).__init__()
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.evaluate_interval = args.evaluate_interval
        dataset = build_dataset(args) if dataset is None else dataset
        self.data = dataset[0]
        self.data.apply(lambda x: x.to(self.device))
        args.num_entities = len(torch.unique(self.data.edge_index))
        args.num_rels = len(torch.unique(self.data.edge_attr))
        model = build_model(args) if model is None else model
        self.model = model.to(self.device)
        self.max_epoch = args.max_epoch
        self.patience = min(args.patience, 20)
        self.grad_norm = 1.0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        )

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_mrr = 0
        best_model = None
        val_mrr = 0

        for epoch in epoch_iter:
            loss_n = self._train_step()
            if (epoch + 1) % self.evaluate_interval == 0:
                torch.cuda.empty_cache()
                val_mrr, _ = self._test_step("val")
                if val_mrr > best_mrr:
                    best_mrr = val_mrr
                    best_model = copy.deepcopy(self.model)
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        self.model = best_model
                        epoch_iter.close()
                        break
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, TrainLoss: {loss_n: .4f}, Val MRR: {val_mrr: .4f}, Best MRR: {best_mrr: .4f}"
            )
        self.model = best_model
        test_mrr, test_hits = self._test_step("test")
        print(
            f"Test MRR:{test_mrr}, Hits@1/3/10: {test_hits}"
        )
        return dict(MRR=test_mrr, HITS1=test_hits[0], HITS3=test_hits[1], HITS10=test_hits[2])

    def _train_step(self, split="train"):
        self.model.train()
        self.optimizer.zero_grad()
        loss_n = self.model.loss(self.data)
        loss_n.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        return loss_n.item()
    
    def _test_step(self, split="val"):
        self.model.eval()
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        edge_index = self.data.edge_index[:, mask]
        edge_attr = self.data.edge_attr[mask]
        mrr, hits = self.model.predict(edge_index, edge_attr)
        return mrr, hits

@register_task("link_prediction")
class LinkPrediction(BaseTask):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--evaluate-interval", type=int, default=30)
        parser.add_argument("--max-epoch", type=int, default=3000)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight-decay", type=float, default=0)
        
        parser.add_argument("--hidden-size", type=int, default=200) # KG
        parser.add_argument("--negative-ratio", type=int, default=5)
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(LinkPrediction, self).__init__(args)
        if select_task(args.model, model):
            self.task = KGLinkPrediction(args, dataset, model)
        else:
            self.task = HomoLinkPrediction(args, dataset, model)
    
    def train(self):
        return self.task.train()
