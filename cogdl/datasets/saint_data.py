import json
import torch
import numpy as np
import os.path as osp
import scipy.sparse as sp

from cogdl.data import Data, Dataset
from cogdl.data import download_url
from cogdl.utils import multilabel_evaluator
from .planetoid_data import index_to_mask, normalize_feature
from . import register_dataset


def read_saint_data(folder):
    names = ["adj_full.npz", "adj_train.npz", "class_map.json", "feats.npy", "role.json"]
    names = [osp.join(folder, name) for name in names]
    adj_full = sp.load_npz(names[0])
    adj_train = sp.load_npz(names[1])
    class_map = json.load(open(names[2]))
    feats = np.load(names[3])
    role = json.load(open(names[4]))

    train_mask = index_to_mask(role["tr"], size=feats.shape[0])
    val_mask = index_to_mask(role["va"], size=feats.shape[0])
    test_mask = index_to_mask(role["te"], size=feats.shape[0])

    feats = torch.from_numpy(feats).float()
    item = class_map["0"]
    label_matrix = np.zeros((feats.shape[0], len(item)), dtype=float)
    for key, val in class_map.items():
        label_matrix[int(key)] = np.array(val)
    label_matrix = torch.from_numpy(label_matrix)

    def get_adj(adj):
        row, col = adj.nonzero()
        data = adj.data
        row = torch.tensor(row, dtype=torch.long)
        col = torch.tensor(col, dtype=torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = torch.tensor(data, dtype=torch.float)
        return edge_index, edge_attr

    edge_index_full, edge_attr_full = get_adj(adj_full)
    edge_index_train, edge_attr_train = get_adj(adj_train)

    data = Data(
        x=feats,
        y=label_matrix,
        edge_index=edge_index_full,
        edge_attr=edge_attr_full,
        edge_index_train=edge_index_train,
        edge_attr_train=edge_attr_train,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data


class SAINTDataset(Dataset):

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    def __init__(self, root, name):
        self.name = name

        super(SAINTDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ["adj_full.npz", "adj_train.npz", "class_map.json", "feats.npy", "role.json"]
        return names

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_classes(self):
        assert hasattr(self.data, "y")
        return self.data.y.shape[1]

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}".format(self.url, name), self.raw_dir)

    def process(self):
        data = read_saint_data(self.raw_dir)
        torch.save(data, self.processed_paths[0])

    def get(self, idx):
        return self.data

    def get_evaluator(self):
        return multilabel_evaluator()

    def __repr__(self):
        return "{}()".format(self.name)

    def __len__(self):
        return self.data.x.shape[0]


@register_dataset("yelp")
class YelpDataset(SAINTDataset):
    def __init__(self, args=None):
        dataset = "Yelp"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset)
        super(YelpDataset, self).__init__(path, dataset)
        self.data = normalize_feature(self.data)


@register_dataset("amazon-s")
class AmazonDataset(SAINTDataset):
    def __init__(self, args=None):
        dataset = "AmazonSaint"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset)
        super(AmazonDataset, self).__init__(path, dataset)
        self.data = normalize_feature(self.data)
