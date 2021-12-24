import torch
import networkx as nx
import numpy as np
import os.path as osp
import pickle as pkl
import sys
import scipy.sparse as sp

from cogdl.data import Dataset, Graph
from cogdl.utils import download_url, untar, Accuracy, CrossEntropyLoss


def read_geom_data(folder, dataset_name):
    graph_adjacency_list_file_path = osp.join(folder, "out1_graph_edges.txt")
    graph_node_features_and_labels_file_path = osp.join(folder, "out1_node_feature_label.txt")

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == "film":
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split("\t")
                assert len(line) == 3
                assert int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(","), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split("\t")
                assert len(line) == 3
                assert int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(","), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split("\t")
            assert len(line) == 2
            if int(line[0]) not in G:
                G.add_node(
                    int(line[0]), features=graph_node_features_dict[int(line[0])], label=graph_labels_dict[int(line[0])]
                )
            if int(line[1]) not in G:
                G.add_node(
                    int(line[1]), features=graph_node_features_dict[int(line[1])], label=graph_labels_dict[int(line[1])]
                )
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array([features for _, features in sorted(G.nodes(data="features"), key=lambda x: x[0])])
    labels = np.array([label for _, label in sorted(G.nodes(data="label"), key=lambda x: x[0])])

    all_masks = []
    for split in range(10):
        graph_split_file_path = osp.join(folder, f"{dataset_name}_split_0.6_0.2_{split}.npz")
        with np.load(graph_split_file_path) as splits_file:
            train_mask = splits_file["train_mask"]
            val_mask = splits_file["val_mask"]
            test_mask = splits_file["test_mask"]
            train_mask = torch.BoolTensor(train_mask)
            val_mask = torch.BoolTensor(val_mask)
            test_mask = torch.BoolTensor(test_mask)
            all_masks.append({"train": train_mask, "val": val_mask, "test": test_mask})

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    coo_adj = adj.tocoo()
    row, col = torch.LongTensor(coo_adj.row), torch.LongTensor(coo_adj.col)
    edge_index = (row, col)

    data = Graph(x=features, edge_index=edge_index, y=labels, all_masks=all_masks)

    return data


class GeomDataset(Dataset):
    url = "https://cloud.tsinghua.edu.cn/d/70d8aaebf2ed493697e0/files/?p=%2F"

    def __init__(self, root, name, split=0):
        self.name = name
        self.split = split

        super(GeomDataset, self).__init__(root)

        self.data = torch.load(self.processed_paths[0])
        self.raw_dir = osp.join(self.root, self.name, "raw")
        self.processed_dir = osp.join(self.root, self.name, "processed")

        self.data.train_mask = self.data.all_masks[split]["train"]
        self.data.val_mask = self.data.all_masks[split]["val"]
        self.data.test_mask = self.data.all_masks[split]["test"]

        self.data.set_asymmetric()

    @property
    def raw_file_names(self):
        names = ["out1_graph_edges.txt", "out1_node_feature_label.txt"] + [
            f"{self.name}_split_0.6_0.2_{idx}.npz" for idx in range(10)
        ]
        return names

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_classes(self):
        assert hasattr(self.data, "y")
        return int(torch.max(self.data.y)) + 1

    @property
    def num_nodes(self):
        assert hasattr(self.data, "y")
        return self.data.y.shape[0]

    def download(self):
        fname = "{}.zip".format(self.name.lower())
        download_url("{}{}.zip&dl=1".format(self.url, self.name.lower()), self.raw_dir, fname)
        untar(self.raw_dir, fname)

    def process(self):
        data = read_geom_data(self.raw_dir, self.name)
        torch.save(data, self.processed_paths[0])

        return data

    def get(self, idx):
        return self.data

    def __repr__(self):
        return "{}()".format(self.name)

    def __len__(self):
        return 1

    def get_evaluator(self):
        return Accuracy()

    def get_loss_fn(self):
        return CrossEntropyLoss()


class ChameleonDataset(GeomDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "chameleon"
        path = osp.join(data_path, dataset)
        super(ChameleonDataset, self).__init__(path, dataset, split)


class CornellDataset(GeomDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "cornell"
        path = osp.join(data_path, dataset)
        super(CornellDataset, self).__init__(path, dataset, split)


class FilmDataset(GeomDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "film"
        path = osp.join(data_path, dataset)
        super(FilmDataset, self).__init__(path, dataset, split)


class SquirrelDataset(GeomDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "squirrel"
        path = osp.join(data_path, dataset)
        super(SquirrelDataset, self).__init__(path, dataset, split)


class TexasDataset(GeomDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "texas"
        path = osp.join(data_path, dataset)
        super(TexasDataset, self).__init__(path, dataset, split)


class WisconsinDataset(GeomDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "wisconsin"
        path = osp.join(data_path, dataset)
        super(WisconsinDataset, self).__init__(path, dataset, split)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def read_planetoid_data(folder, dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open(osp.join(folder, "ind.{}.{}".format(dataset_str, names[i])), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(osp.join(folder, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    all_masks = []
    for split in range(10):
        graph_split_file_path = osp.join(folder, f"{dataset_str}_split_0.6_0.2_{split}.npz")
        with np.load(graph_split_file_path) as splits_file:
            train_mask = splits_file["train_mask"]
            val_mask = splits_file["val_mask"]
            test_mask = splits_file["test_mask"]
            train_mask = torch.BoolTensor(train_mask)
            val_mask = torch.BoolTensor(val_mask)
            test_mask = torch.BoolTensor(test_mask)
            all_masks.append({"train": train_mask, "val": val_mask, "test": test_mask})

    features = torch.FloatTensor(features.todense())
    labels = np.argmax(labels, axis=-1)
    labels = torch.LongTensor(labels)

    coo_adj = adj.tocoo()
    row, col = torch.LongTensor(coo_adj.row), torch.LongTensor(coo_adj.col)
    edge_index = (row, col)

    graph = Graph(x=features, edge_index=edge_index, y=labels, all_masks=all_masks)

    return graph


class GeomPlanetoidDataset(Dataset):
    url = "https://cloud.tsinghua.edu.cn/d/70d8aaebf2ed493697e0/files/?p=%2F"

    def __init__(self, root, name, split=0):
        self.name = name
        self.split = split

        super(GeomPlanetoidDataset, self).__init__(root)

        self.data = torch.load(self.processed_paths[0])
        self.raw_dir = osp.join(self.root, self.name, "raw")
        self.processed_dir = osp.join(self.root, self.name, "processed")

        self.data.train_mask = self.data.all_masks[split]["train"]
        self.data.val_mask = self.data.all_masks[split]["val"]
        self.data.test_mask = self.data.all_masks[split]["test"]

    @property
    def raw_file_names(self):
        tmp_name = self.name.lower()[:-5]
        names = [
            "ind.{}.{}".format(tmp_name, name) for name in ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        ] + [f"{tmp_name}_split_0.6_0.2_{idx}.npz" for idx in range(10)]
        return names

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_classes(self):
        assert hasattr(self.data, "y")
        return int(torch.max(self.data.y)) + 1

    @property
    def num_nodes(self):
        assert hasattr(self.data, "y")
        return self.data.y.shape[0]

    def download(self):
        fname = "{}.zip".format(self.name.lower())
        download_url("{}{}.zip&dl=1".format(self.url, self.name.lower()), self.raw_dir, fname)
        untar(self.raw_dir, fname)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name[:-5])
        torch.save(data, self.processed_paths[0])

        return data

    def get(self, idx):
        return self.data

    def __repr__(self):
        return "{}()".format(self.name)

    def __len__(self):
        return 1

    def get_evaluator(self):
        return Accuracy()

    def get_loss_fn(self):
        return CrossEntropyLoss()


class CoraGeomDataset(GeomPlanetoidDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "cora_geom"
        path = osp.join(data_path, dataset)
        super(CoraGeomDataset, self).__init__(path, dataset, split)


class CiteSeerGeomDataset(GeomPlanetoidDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "citeseer_geom"
        path = osp.join(data_path, dataset)
        super(CiteSeerGeomDataset, self).__init__(path, dataset, split)


class PubMedGeomDataset(GeomPlanetoidDataset):
    def __init__(self, data_path="data", split=0):
        dataset = "pubmed_geom"
        path = osp.join(data_path, dataset)
        super(PubMedGeomDataset, self).__init__(path, dataset, split)
