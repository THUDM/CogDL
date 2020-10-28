import json
import os
import os.path as osp
from itertools import product

import numpy as np
import scipy.io
import torch
from collections import defaultdict

from cogdl.data import Data, Dataset, download_url

from . import register_dataset


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

        data = Data(edge_index=torch.LongTensor(edge_list).t(), x=None, y=y)

        torch.save(data, self.processed_paths[0])


@register_dataset("usa-airport")
class USAAirportDataset(Edgelist):
    def __init__(self):
        dataset = "usa-airport"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(USAAirportDataset, self).__init__(path, dataset)
