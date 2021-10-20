import random
import networkx as nx
import numpy as np
import torch

from .. import DataWrapper
from cogdl.data import Graph


class EmbeddingLinkPredictionDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--negative-ratio", type=int, default=5)
        # fmt: on

    def __init__(self, dataset, negative_ratio):
        super(EmbeddingLinkPredictionDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.negative_ratio = negative_ratio
        self.train_data, self.test_data = None, None

    def train_wrapper(self):
        return self.train_data

    def test_wrapper(self):
        return self.test_data

    def pre_transform(self):
        row, col = self.dataset.data.edge_index
        edge_list = list(zip(row.numpy(), col.numpy()))
        edge_set = set()
        for edge in edge_list:
            if (edge[0], edge[1]) not in edge_set and (edge[1], edge[0]) not in edge_set:
                edge_set.add(edge)
        edge_list = list(edge_set)
        train_edges, test_edges = divide_data(edge_list, [0.90, 0.10])
        self.test_data = gen_node_pairs(train_edges, test_edges, self.negative_ratio)
        train_edges = np.array(train_edges).transpose()
        train_edges = torch.from_numpy(train_edges)
        self.train_data = Graph(edge_index=train_edges)


def divide_data(input_list, division_rate):
    local_division = len(input_list) * np.cumsum(np.array(division_rate))
    random.shuffle(input_list)
    return [
        input_list[int(round(local_division[i - 1])) if i > 0 else 0 : int(round(local_division[i]))]
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
    test_false_data = randomly_choose_false_edges(list(training_nodes), train_data, len(test_data) * negative_ratio)
    return (test_true_data, test_false_data)


def get_score(embs, node1, node2):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
