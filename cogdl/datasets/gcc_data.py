import os
import os.path as osp
import numpy as np
from collections import defaultdict

import torch

from cogdl.data import Graph, Dataset
from cogdl.utils import download_url

from . import register_dataset


class GCCDataset(Dataset):
    url = "https://github.com/cenyk1230/gcc-data/raw/master"

    def __init__(self, root, name):
        self.name = name
        super(GCCDataset, self).__init__(root)

        name1 = name.split("_")[0]
        name2 = name.split("_")[1]
        edge_index_1, dict_1, self.node2id_1 = self.preprocess(root, name1)
        edge_index_2, dict_2, self.node2id_2 = self.preprocess(root, name2)
        self.data = [
            Graph(x=None, edge_index=edge_index_1, y=dict_1),
            Graph(x=None, edge_index=edge_index_2, y=dict_2),
        ]
        self.transform = None

    @property
    def raw_file_names(self):

        names = [
            self.name.split("_")[0] + ".dict",
            self.name.split("_")[0] + ".graph",
            self.name.split("_")[1] + ".dict",
            self.name.split("_")[1] + ".graph",
        ]
        return names

    @property
    def processed_file_names(self):
        return []

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}/{}".format(self.url, self.name.lower(), name), self.raw_dir)

    def preprocess(self, root, name):
        dict_path = os.path.join(root, "raw/" + name + ".dict")
        graph_path = os.path.join(root, "raw/" + name + ".graph")

        with open(graph_path, "r") as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.strip().split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        name_dict = dict()
        with open(dict_path) as f:
            for line in f:
                name, str_x = line.split("\t")
                x = int(str_x)
                if x not in node2id:
                    node2id[x] = len(node2id)
                name_dict[name] = node2id[x]

        return torch.LongTensor(edge_list).t(), name_dict, node2id

    def __repr__(self):
        return "{}()".format(self.name)


class Edgelist(Dataset):
    url = "https://github.com/cenyk1230/gcc-data/raw/master"

    def __init__(self, root, name):
        self.name = name
        super(Edgelist, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ["edgelist.txt", "nodelabel.txt"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def num_classes(self):
        return self.data.y.shape[1]

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}/{}".format(self.url, self.name.lower(), name), self.raw_dir)

    def get(self, idx):
        assert idx == 0
        return self.data

    def process(self):
        edge_list_path = osp.join(self.raw_dir, "edgelist.txt")
        node_label_path = osp.join(self.raw_dir, "nodelabel.txt")

        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label])
            if "hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1

        data = Graph(edge_index=torch.LongTensor(edge_list).t(), x=None, y=y)

        torch.save(data, self.processed_paths[0])


@register_dataset("kdd_icdm")
class KDD_ICDM_GCCDataset(GCCDataset):
    def __init__(self, data_path="data"):
        dataset = "kdd_icdm"
        path = osp.join(data_path, dataset)
        super(KDD_ICDM_GCCDataset, self).__init__(path, dataset)


@register_dataset("sigir_cikm")
class SIGIR_CIKM_GCCDataset(GCCDataset):
    def __init__(self, data_path="data"):
        dataset = "sigir_cikm"
        path = osp.join(data_path, dataset)
        super(SIGIR_CIKM_GCCDataset, self).__init__(path, dataset)


@register_dataset("sigmod_icde")
class SIGMOD_ICDE_GCCDataset(GCCDataset):
    def __init__(self, data_path="data"):
        dataset = "sigmod_icde"
        path = osp.join(data_path, dataset)
        super(SIGMOD_ICDE_GCCDataset, self).__init__(path, dataset)


@register_dataset("usa-airport")
class USAAirportDataset(Edgelist):
    def __init__(self, data_path="data"):
        dataset = "usa-airport"
        path = osp.join(data_path, dataset)
        super(USAAirportDataset, self).__init__(path, dataset)
