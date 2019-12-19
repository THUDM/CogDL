import json
import os
import os.path as osp
import sys
from itertools import product

import networkx as nx
import numpy as np
import torch

from cogdl.data import Data, Dataset, download_url

from . import register_dataset


def read_edgelist_label_data(folder, prefix):
    graph_path = osp.join(folder, "{}.ungraph".format(prefix))
    cmty_path = osp.join(folder, "{}.cmty".format(prefix))

    G = nx.read_edgelist(graph_path, nodetype=int, create_using=nx.Graph())
    num_node = G.number_of_nodes()
    print("edge number: ", num_node)
    with open(graph_path) as f:
        context = f.readlines()
        print("edge number: ", len(context))
        edge_index = np.zeros((2, len(context)))
        for i, line in enumerate(context):
            edge_index[:, i] = list(map(int, line.strip().split("\t")))
    edge_index = torch.from_numpy(edge_index).to(torch.int)

    with open(cmty_path) as f:
        context = f.readlines()
        print("class number: ", len(context))
        label = np.zeros((num_node, len(context)))

        for i, line in enumerate(context):
            line = map(int, line.strip().split("\t"))
            for node in line:
                label[node, i] = 1

    y = torch.from_numpy(label).to(torch.float)
    data = Data(x=None, edge_index=edge_index, y=y)

    return data


class EdgelistLabel(Dataset):
    r"""networks from the https://github.com/THUDM/ProNE/raw/master/data

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wikipedia"`).
    """

    url = "https://github.com/THUDM/ProNE/raw/master/data"

    def __init__(self, root, name):
        self.name = name
        super(EdgelistLabel, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        splits = [self.name]
        files = ["ungraph", "cmty"]
        return ["{}.{}".format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}".format(self.url, name), self.raw_dir)

    def process(self):
        data = read_edgelist_label_data(self.raw_dir, self.name)
        torch.save(data, self.processed_paths[0])


@register_dataset("dblp")
class DBLP(EdgelistLabel):
    def __init__(self):
        dataset = "dblp"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(DBLP, self).__init__(path, dataset)
