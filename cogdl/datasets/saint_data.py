import json
import torch
import numpy as np
import os.path as osp
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from cogdl.data import Data, Dataset
from cogdl.utils import multilabel_evaluator, download_url
from .planetoid_data import index_to_mask
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
    def __init__(self, root, name, url=None):
        self.name = name
        self.url = url
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
            download_url(self.url.format(name), self.raw_dir, name=name)

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


def scale_feats(data):
    scaler = StandardScaler()
    scaler.fit(data.x.numpy())
    data.x = torch.from_numpy(scaler.transform(data.x)).float()
    return data


@register_dataset("yelp")
class YelpDataset(SAINTDataset):
    def __init__(self, args=None):
        dataset = "Yelp"
        url = "https://cloud.tsinghua.edu.cn/d/7218cc013c9a40159306/files/?p=%2F{}&dl=1"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(YelpDataset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)


@register_dataset("amazon-s")
class AmazonDataset(SAINTDataset):
    def __init__(self, args=None):
        dataset = "AmazonSaint"
        url = "https://cloud.tsinghua.edu.cn/d/ae4b2c4f59bd41be9b0b/files/?p=%2F{}&dl=1"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(AmazonDataset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)
