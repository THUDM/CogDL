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

from cognitive_graph import options
from cognitive_graph.datasets import build_dataset
from cognitive_graph.models import build_model

from . import BaseTask, register_task


class NEG_loss(nn.Module):
    def __init__(self, num_nodes, num_sampled, degree=None):
        super(NEG_loss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        if degree is not None:
            self.weights = F.normalize(torch.Tensor(degree).pow(0.75), dim=0)
        else:
            self.weights = torch.ones((num_nodes,), dtype=torch.float) / num_nodes

    def forward(self, input, embs):
        u, v = input
        n = u.shape[0]
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(embs[u], embs[v]), 1)))
        negs = torch.multinomial(
            self.weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(embs[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs[u].unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


class RWGraph:
    def __init__(self, nx_G, alpha=0.0):
        self.G = nx_G
        self.alpha = alpha

    def walk(self, walk_length, start):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()

        if start:
            walk = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            walk = [rand.choice(list(G.nodes()))]

        while len(walk) < walk_length:
            cur = walk[-1]
            if len(G[cur]) > 0:
                if rand.random() >= self.alpha:
                    walk.append(rand.choice(list(G[cur].keys())))
                else:
                    walk.append(walk[0])
            else:
                break
        return [str(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(walk_length=walk_length, start=node))

        return walks


def generate_pairs(walks, vocab):
    pairs = []
    skip_window = 2
    for walk in walks:
        for i in range(len(walk)):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index))
                if i + j < len(walk):
                    pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index))
    return pairs


def generate_vocab(walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for walk in walks:
        for word in walk:
            raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


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


def gen_node_pairs(edge_list):
    edge_list = edge_list.cpu().numpy()
    edge_list = list(zip(edge_list[0], edge_list[1]))
    train_data, valid_data, test_data = divide_data(edge_list, [0.85, 0.05, 0.10])

    G = nx.Graph()
    G.add_edges_from(train_data)
    RWG = RWGraph(G)

    base_walks = RWG.simulate_walks(20, 10)
    vocab, index2word = generate_vocab(base_walks)
    train_pairs = generate_pairs(base_walks, vocab)

    training_nodes = set(list(G.nodes()))
    valid_true_data = []
    test_true_data = []
    for u, v in valid_data:
        if u in training_nodes and v in training_nodes:
            valid_true_data.append((u, v))
    for u, v in test_data:
        if u in training_nodes and v in training_nodes:
            test_true_data.append((u, v))
    valid_false_data = randomly_choose_false_edges(
        list(training_nodes), train_data, len(valid_data)
    )
    test_false_data = randomly_choose_false_edges(
        list(training_nodes), train_data, len(test_data)
    )
    return (
        np.array(train_pairs).T,
        (valid_true_data, valid_false_data),
        (test_true_data, test_false_data),
    )


def get_score(embs, node1, node2):
    vector1 = embs[int(node1)].cpu().detach().numpy()
    vector2 = embs[int(node2)].cpu().detach().numpy()
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
        parser.add_argument("--negative-ratio", type=int, default=5)
        parser.add_argument("--patience", type=int, default=5)
        # fmt: on

    def __init__(self, args):
        super(LinkPrediction, self).__init__(args)

        dataset = build_dataset(args)
        data = dataset[0]
        self.data = data.cuda()
        args.num_features = dataset.num_features
        model = build_model(args)
        self.model = model.cuda()
        self.patience = args.patience

        self.train_data, self.valid_data, self.test_data = gen_node_pairs(self.data.edge_index)

        edge_list = self.data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_list.T.tolist())
        self.criterion = NEG_loss(
            len(self.data.x),
            args.negative_ratio,
            degree=np.array(list(dict(G.degree()).values())),
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def train(self, num_epoch):
        epoch_iter = tqdm(range(num_epoch))
        patience = 0
        best_score = 0
        for epoch in epoch_iter:
            self._train_step()
            roc_auc, f1_score, pr_auc = self._test_step(self.valid_data)
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, ROC-AUC: {roc_auc:.4f}, F1: {f1_score:.4f}, PR-AUC: {pr_auc:.4f}"
            )
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = copy.deepcopy(self.model)
                patience = 0
            else:
                patience += 1
                if patience > self.patience:
                    self.model = best_model
                    break
        roc_auc, f1_score, pr_auc = self._test_step(self.test_data)
        print(
            f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}"
        )
        return roc_auc, f1_score, pr_auc

    def _train_step(self):
        self.model.train()
        embs = self.model(self.data.x, self.data.edge_index)

        self.optimizer.zero_grad()
        self.criterion(self.train_data, embs).backward()
        self.optimizer.step()

    def _test_step(self, test_data):
        self.model.eval()
        embs = self.model(self.data.x, self.data.edge_index)
        roc_auc, f1_score, pr_auc = evaluate(embs, test_data[0], test_data[1])
        return roc_auc, f1_score, pr_auc
