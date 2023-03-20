import functools
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from cogdl.utils import split_dataset_general

from .. import BaseModel


class PatchySAN(BaseModel):
    r"""The Patchy-SAN model from the `"Learning Convolutional Neural Networks for Graphs"
    <https://arxiv.org/abs/1605.05273>`_ paper.

    Args:
        batch_size (int) : The batch size of training.
        sample (int) : Number of chosen vertexes.
        stride (int) : Node selection stride.
        neighbor (int) : The number of neighbor for each node.
        iteration (int) : The number of training iteration.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-sample", default=30, type=int, help="Number of chosen vertexes")
        # parser.add_argument("--stride", default=1, type=int, help="Stride of chosen vertexes")
        parser.add_argument("--num-neighbor", default=10, type=int, help="Number of neighbor in constructing features")
        parser.add_argument("--iteration", default=5, type=int, help="Number of iteration")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.num_sample,
            # args.stride,
            args.num_neighbor,
            args.iteration,
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        # process each graph and add it into Data() as attribute x
        for i, data in enumerate(dataset):
            new_feature = get_single_feature(
                dataset[i], args.num_features, args.num_classes, args.num_sample, args.num_neighbor, args.stride
            )
            dataset[i].x = torch.from_numpy(new_feature)

        return split_dataset_general(dataset, args)

    # def __init__(self, batch_size, num_features, num_classes, num_sample, stride, num_neighbor, iteration):
    def __init__(self, num_features, num_classes, num_sample, num_neighbor, iteration):
        super(PatchySAN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_sample = num_sample
        self.num_neighbor = num_neighbor
        self.iteration = iteration

        # self.build_model(self.num_features, self.num_sample, self.num_neighbor, self.num_classes)
        self.build_model(self.num_features, self.num_sample, self.num_neighbor, self.num_classes)

    def build_model(self, num_channel, num_sample, num_neighbor, num_class):
        rep1, stride1 = 4, 4
        num_filter1, num_filter2 = 16, 8
        self.conv1 = nn.Conv1d(num_channel, num_filter1, rep1, stride=stride1, groups=1)
        self.conv2 = nn.Conv1d(num_filter1, num_filter2, num_neighbor, stride=1, groups=1)

        num_lin = (int(num_sample * num_neighbor / stride1) - num_neighbor + 1) * num_filter2
        self.lin1 = torch.nn.Linear(num_lin, 128)
        self.lin2 = torch.nn.Linear(128, num_class)

        self.nn = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            self.lin1,
            nn.ReLU(),
            nn.Dropout(0.2),
            self.lin2,
            nn.Softmax(),
        )

    def forward(self, batch):
        logits = self.nn(batch.x)
        return logits


def assemble_neighbor(G, node, num_neighbor, sorted_nodes):
    """assemble neighbors for node with BFS strategy"""
    neighbors_dict = dict()
    new_neighbors_dict = dict()
    neighbors_dict[node] = 0
    new_neighbors_dict[node] = 0
    # assemble over K neighbors with BFS strategy
    while len(neighbors_dict) < num_neighbor and len(new_neighbors_dict) > 0:
        temp_neighbor_dict = dict()
        for v, d in new_neighbors_dict.items():
            for new_v in G.neighbors(v):
                if new_v not in temp_neighbor_dict:
                    temp_neighbor_dict[new_v] = d + 1
        n = len(neighbors_dict)
        for v, d in temp_neighbor_dict.items():
            if v not in neighbors_dict:
                neighbors_dict[v] = d
        new_neighbors_dict = temp_neighbor_dict
        # break if the number of neighbors do not increase
        if n == len(neighbors_dict):
            break

    # add dummy disconnected nodes if number is not suffice
    while len(neighbors_dict) < num_neighbor:
        rand_node = sorted_nodes[random.randint(0, len(sorted_nodes) - 1)][0]
        if rand_node not in neighbors_dict:
            neighbors_dict[rand_node] = 10
    return neighbors_dict


def cmp(s1, s2):
    list1 = [int(l) for l in s1.strip().split(" ")]  # noqa E741
    list2 = [int(l) for l in s2.strip().split(" ")]  # noqa E741
    i = 0
    while i < len(list1) and i < len(list2):
        if list1[i] < list2[i]:
            return -1
        if list1[i] > list2[i]:
            return 1
        i += 1
    if i < len(list1):
        return 1
    elif i < len(list2):
        return -1
    else:
        return 0


def one_dim_wl(graph_list, init_labels, iteration=5):
    """1-dimension Wl method used for node normalization for all the subgraphs"""
    sorted_labels = sorted(list(set(init_labels.values())))
    label_dict = dict([(label, index) for index, label in enumerate(sorted_labels)])
    graph_label_list = []
    for t in range(iteration):
        new_label_dict = dict()
        # get label for each node in each graph
        if t == 0:
            # get label according to nodes' attribute labels
            for i in range(len(graph_list)):
                labels = dict()
                for id, v in enumerate(graph_list[i].nodes()):
                    labels[v] = str(init_labels[v])
                    new_label_dict[labels[v]] = 1
                graph_label_list.append(labels)
        else:
            # get label according to neighbors' labels
            for i in range(len(graph_list)):
                labels = dict()
                for id, v in enumerate(graph_list[i].nodes()):
                    neighbor_labels = [graph_label_list[i][v2] for v2 in graph_list[i].neighbors(v)]
                    sorted_labels = [str(l) for l in sorted(neighbor_labels)]  # noqa E741
                    # concentrate node label and its sorted neighbors' labels
                    new_label = str(graph_label_list[i][v]) + " " + " ".join(sorted_labels)
                    new_label_dict[new_label] = 1
                    labels[v] = new_label
                graph_label_list[i] = labels.copy()

        # sort new labels with dictionary order
        sorted_list = sorted(new_label_dict.keys(), key=functools.cmp_to_key(cmp), reverse=False)
        for new_label in sorted_list:
            # add new label to the labels_dict
            if new_label not in label_dict:
                label_dict[new_label] = len(label_dict)

        # relabel node labels with new label
        for i in range(len(graph_list)):
            for id, v in enumerate(graph_list[i].nodes()):
                graph_label_list[i][v] = label_dict[graph_label_list[i][v]]
    return graph_label_list


def node_selection_with_1d_wl(G, features, num_channel, num_sample, num_neighbor, stride):
    """construct features for cnn"""
    X = np.zeros((num_channel, num_sample, num_neighbor), dtype=np.float32)
    node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])
    betweenness = nx.betweenness_centrality(G)
    sorted_nodes = sorted(betweenness.items(), key=lambda d: d[1], reverse=False)
    # obtain normalized neighbors' features for each vertex
    i = 0
    j = 0
    root_list = []
    distance_list = []
    graph_list = []
    while j < num_sample:
        if i < len(sorted_nodes):
            neighbors_dict = assemble_neighbor(G, sorted_nodes[i][0], num_neighbor, sorted_nodes)
            # construct subgraph and sort neighbors with a labeling measurement like degree centrality
            sub_g = G.subgraph(neighbors_dict.keys())
            root_list.append(sorted_nodes[i][0])
            distance_list.append(neighbors_dict)
            graph_list.append(sub_g)
        else:
            # zero receptive field
            X[:, j, :] = np.zeros((num_channel, num_neighbor), dtype=np.float32)
        i += stride
        j += 1
    init_labels = dict([(v, features[id].argmax(axis=0)) for v, id in node2id.items()])
    graph_labels_list = one_dim_wl(graph_list, init_labels)

    # finally relabel based on 1d-wl and distance to root node
    for i in range(len(root_list)):
        # set root node the first position
        graph_labels_list[i][root_list[i]] = 0
        sorted_measurement = dict([(v, [measure, distance_list[i][v]]) for v, measure in graph_labels_list[i].items()])
        sorted_neighbor = sorted(sorted_measurement.items(), key=lambda d: d[1], reverse=False)[:num_neighbor]
        reorder_dict = dict(zip(sorted(sorted_measurement.keys()), range(len(sorted_measurement))))
        X[:, i, :] = features[[reorder_dict[v] for v, measure in sorted_neighbor]].T
    return X.reshape(num_channel, num_sample * num_neighbor)


def get_single_feature(data, num_features, num_classes, num_sample, num_neighbor, stride=1):
    """construct features"""
    data_list = [data]
    X = np.zeros((len(data_list), num_features, num_sample * num_neighbor), dtype=np.float32)
    for i in range(len(data_list)):
        edge_index, features = data_list[i].edge_index, data_list[i].x
        G = nx.Graph()
        row, col = edge_index[0].numpy(), edge_index[1].numpy()
        G.add_edges_from(list(zip(row, col)))
        # print("graph", i, "number of node", G.number_of_nodes(), "edge", G.number_of_edges())
        if G.number_of_nodes() > num_neighbor:
            X[i] = node_selection_with_1d_wl(G, features.cpu().numpy(), num_features, num_sample, num_neighbor, stride)
    X = X.astype(np.float32)
    return X
