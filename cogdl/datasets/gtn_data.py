import os.path as osp
import pickle

import numpy as np
from cogdl.data import Graph, Dataset
from cogdl.utils import download_url, untar
from cogdl import function as BF

class GTNDataset(Dataset):
    r"""The network datasets "ACM", "DBLP" and "IMDB" from the
    `"Graph Transformer Networks"
    <https://arxiv.org/abs/1911.06455>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"gtn-acm"`,
            :obj:`"gtn-dblp"`, :obj:`"gtn-imdb"`).
    """

    def __init__(self, root, name):
        self.name = name
        self.url = f"https://github.com/cenyk1230/gtn-data/blob/master/{name}.zip?raw=true"
        super(GTNDataset, self).__init__(root)
        self.data = BF.load(self.processed_paths[0])
        self.num_edge = len(self.data.adj)
        self.num_nodes = self.data.x.shape[0]

    @property
    def raw_file_names(self):
        names = ["edges.pkl", "labels.pkl", "node_features.pkl"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def num_classes(self):
        return BF.max(self.data.train_target).item() + 1

    def read_gtn_data(self, folder):
        edges = pickle.load(open(osp.join(folder, "edges.pkl"), "rb"))
        labels = pickle.load(open(osp.join(folder, "labels.pkl"), "rb"))
        node_features = pickle.load(open(osp.join(folder, "node_features.pkl"), "rb"))

        data = Graph()
        data.x = BF.from_numpy(node_features).float()

        num_nodes = edges[0].shape[0]

        node_type = np.zeros((num_nodes), dtype=int)
        assert len(edges) == 4
        assert len(edges[0].nonzero()) == 2

        node_type[edges[0].nonzero()[0]] = 0
        node_type[edges[0].nonzero()[1]] = 1
        node_type[edges[1].nonzero()[0]] = 1
        node_type[edges[1].nonzero()[1]] = 0
        node_type[edges[2].nonzero()[0]] = 0
        node_type[edges[2].nonzero()[1]] = 2
        node_type[edges[3].nonzero()[0]] = 2
        node_type[edges[3].nonzero()[1]] = 0

        print(node_type)
        data.pos = BF.from_numpy(node_type)

        edge_list = []
        for i, edge in enumerate(edges):
            edge_tmp = BF.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).long()
            edge_list.append(edge_tmp)
        data.edge_index = BF.cat(edge_list, 1)

        A = []
        for i, edge in enumerate(edges):
            edge_tmp = BF.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).long()
            value_tmp = BF.ones(edge_tmp.shape[1]).float()
            A.append((edge_tmp, value_tmp))
        edge_tmp = BF.stack((BF.arange(0, num_nodes), BF.arange(0, num_nodes))).long()
        value_tmp = BF.ones(num_nodes).float()
        A.append((edge_tmp, value_tmp))
        data.adj = A

        data.train_node = BF.from_numpy(np.array(labels[0])[:, 0]).long()
        data.train_target = BF.from_numpy(np.array(labels[0])[:, 1]).long()
        data.valid_node = BF.from_numpy(np.array(labels[1])[:, 0]).long()
        data.valid_target = BF.from_numpy(np.array(labels[1])[:, 1]).long()
        data.test_node = BF.from_numpy(np.array(labels[2])[:, 0]).long()
        data.test_target = BF.from_numpy(np.array(labels[2])[:, 1]).long()

        y = np.zeros((num_nodes), dtype=int)
        x_index = BF.cat((data.train_node, data.valid_node, data.test_node))
        y_index = BF.cat((data.train_target, data.valid_target, data.test_target))
        y[x_index.numpy()] = y_index.numpy()
        data.y = BF.from_numpy(y)
        self.data = data

    def get(self, idx):
        assert idx == 0
        return self.data

    def apply_to_device(self, device):
        self.data.x = self.data.x.to(device)
        self.data.y = self.data.y.to(device)

        self.data.train_node = self.data.train_node.to(device)
        self.data.valid_node = self.data.valid_node.to(device)
        self.data.test_node = self.data.test_node.to(device)

        self.data.train_target = self.data.train_target.to(device)
        self.data.valid_target = self.data.valid_target.to(device)
        self.data.test_target = self.data.test_target.to(device)

        new_adj = []
        for (t1, t2) in self.data.adj:
            new_adj.append((t1.to(device), t2.to(device)))
        self.data.adj = new_adj

    def download(self):
        download_url(self.url, self.raw_dir, name=self.name + ".zip")
        untar(self.raw_dir, self.name + ".zip")

    def process(self):
        self.read_gtn_data(self.raw_dir)
        BF.save(self.data, self.processed_paths[0])

    def __repr__(self):
        return "{}".format(self.name)


class ACM_GTNDataset(GTNDataset):
    def __init__(self, data_path="data"):
        dataset = "gtn-acm"
        path = osp.join(data_path, dataset)
        super(ACM_GTNDataset, self).__init__(path, dataset)


class DBLP_GTNDataset(GTNDataset):
    def __init__(self, data_path="data"):
        dataset = "gtn-dblp"
        path = osp.join(data_path, dataset)
        super(DBLP_GTNDataset, self).__init__(path, dataset)


class IMDB_GTNDataset(GTNDataset):
    def __init__(self, data_path="data"):
        dataset = "gtn-imdb"
        path = osp.join(data_path, dataset)
        super(IMDB_GTNDataset, self).__init__(path, dataset)
