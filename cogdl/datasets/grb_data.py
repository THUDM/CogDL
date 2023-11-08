import os
import os.path as osp
import numpy as np
from collections import defaultdict
import scipy.sparse as sp

import torch

from cogdl.data import Graph, Dataset
from cogdl.utils import untar, download_url
from cogdl.utils.grb_utils import adj_to_tensor


class GRBDataset(Dataset):
    GRB_SUPPORTED_DATASETS = {"grb-cora", "grb-citeseer", "grb-aminer", "grb-reddit", "grb-flickr"}
    MODES = ["easy", "medium", "hard", "full"]
    FEAT_NORMS = [None, "linearize", "arctan", "tanh", "standardize"]
    URLs = {
        "grb-cora": "https://cogdl-data.oss-cn-beijing.aliyuncs.com/data/grb-cora.zip",
        "grb-citeseer": "https://cogdl-data.oss-cn-beijing.aliyuncs.com/data/grb-citeseer.zip",
        "grb-reddit": "https://cogdl-data.oss-cn-beijing.aliyuncs.com/data/grb-reddit.zip",
        "grb-aminer": "https://cogdl-data.oss-cn-beijing.aliyuncs.com/data/grb-aminer.zip",
        "grb-flickr": "https://cogdl-data.oss-cn-beijing.aliyuncs.com/data/grb-flickr.zip",
    }

    def __init__(self, root, name, mode="full", feat_norm=None):
        if name not in self.GRB_SUPPORTED_DATASETS:
            print("{} dataset not supported.".format(name))
            exit(1)
        if mode not in self.MODES:
            print('Mode: {} not supproted. Mode must be choosed from ["easy", "medium", "hard", "full"]'.format(mode))
            exit(1)
        if feat_norm not in self.FEAT_NORMS:
            print(
                'feat_norm: {} not supproted. Feat_norm must be choosed from [None, "linearize", "arctan", "tanh", "standardize"]'.format(
                    feat_norm
                )
            )
            exit(1)
        self.name = name
        self.mode = mode
        self.feat_norm = feat_norm
        super(GRBDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = [
            "adj.npz",
            "features.npz",
            "labels.npz",
            "index.npz",
        ]
        return names

    @property
    def processed_file_names(self):
        return ["data_" + self.mode + "_" + str(self.feat_norm) + ".pt"]

    def download(self):
        fname = "{}.zip".format(self.name.lower())
        download_url(self.URLs[self.name.lower()], self.raw_dir, fname)
        untar(self.raw_dir, fname)

    def get(self, idx):
        assert idx == 0
        return self.data

    def read_grb_data(self, folder, mode, feat_norm):
        adj = sp.load_npz(osp.join(folder, "adj.npz"))
        features = np.load(osp.join(folder, "features.npz")).get("data")
        if feat_norm is not None:
            features = feat_normalize(features, norm=feat_norm)
        num_nodes = features.shape[0]
        labels = np.load(osp.join(folder, "labels.npz")).get("data")
        index = np.load(os.path.join(folder, "index.npz"))
        index_train = index.get("index_train")
        train_mask = torch.zeros(num_nodes, dtype=bool)
        train_mask[index_train] = True
        self.index_train = index_train
        self.train_mask = train_mask

        index_val = index.get("index_val")
        val_mask = torch.zeros(num_nodes, dtype=bool)
        val_mask[index_val] = True
        self.index_val = index_val
        self.val_mask = val_mask

        if mode == "easy":
            index_test = index.get("index_test_easy")
        elif mode == "medium":
            index_test = index.get("index_test_medium")
        elif mode == "hard":
            index_test = index.get("index_test_hard")
        elif mode == "full":
            index_test = index.get("index_test")
        else:
            index_test = index.get("index_test")

        test_mask = torch.zeros(num_nodes, dtype=bool)
        test_mask[index_test] = True
        self.index_test = index_test
        self.test_mask = test_mask
        edge_index, edge_attr = adj2edge(adj)

        data = Graph(
            x=torch.from_numpy(features).type(torch.FloatTensor),
            y=torch.from_numpy(labels),
            grb_adj=adj_to_tensor(adj),
            edge_index=edge_index,
            edge_attr=edge_attr,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return data

    def process(self):
        data = self.read_grb_data(self.raw_dir, self.mode, self.feat_norm)
        torch.save(data, self.processed_paths[0])


class Cora_GRBDataset(GRBDataset):
    def __init__(self, root="data", mode="full", feat_norm=None):
        super(Cora_GRBDataset, self).__init__(osp.join(root, "grb-cora"), "grb-cora", mode, feat_norm)


class Citeseer_GRBDataset(GRBDataset):
    def __init__(self, root="data", mode="full", feat_norm=None):
        super(Citeseer_GRBDataset, self).__init__(osp.join(root, "grb-citeseer"), "grb-citeseer", mode, feat_norm)


class Reddit_GRBDataset(GRBDataset):
    def __init__(self, root="data", mode="full", feat_norm=None):
        super(Reddit_GRBDataset, self).__init__(osp.join(root, "grb-reddit"), "grb-reddit", mode, feat_norm)


class Aminer_GRBDataset(GRBDataset):
    def __init__(self, root="data", mode="full", feat_norm=None):
        super(Aminer_GRBDataset, self).__init__(osp.join(root, "grb-aminer"), "grb-aminer", mode, feat_norm)


class Flickr_GRBDataset(GRBDataset):
    def __init__(self, root="data", mode="full", feat_norm=None):
        super(Flickr_GRBDataset, self).__init__(osp.join(root, "grb-flickr"), "grb-flickr", mode, feat_norm)


def adj2edge(adj: sp.csr.csr_matrix):
    row, col = adj.nonzero()
    data = adj.data
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = torch.tensor(data, dtype=torch.float)
    return edge_index, edge_attr


def feat_normalize(features, norm=None, lim_min=-1.0, lim_max=1.0):
    r"""
    Description
    -----------
    Feature normalization function.

    Parameters
    ----------
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    norm : str, optional
        Type of normalization. Choose from ``["linearize", "arctan", "tanh", "standarize"]``.
        Default: ``None``.
    lim_min : float
        Minimum limit of feature value. Default: ``-1.0``.
    lim_max : float
        Maximum limit of feature value. Default: ``1.0``.

    Returns
    -------
    features : torch.FloatTensor
        Normalized features in form of ``N * D`` torch float tensor.

    """
    if norm == "linearize":
        k = (lim_max - lim_min) / (features.max() - features.min())
        features = lim_min + k * (features - features.min())
    elif norm == "arctan":
        features = (features - features.mean()) / features.std()
        features = 2 * np.arctan(features) / np.pi
    elif norm == "tanh":
        features = (features - features.mean()) / features.std()
        features = np.tanh(features)
    elif norm == "standardize":
        features = (features - features.mean()) / features.std()
    else:
        features = features

    return features
