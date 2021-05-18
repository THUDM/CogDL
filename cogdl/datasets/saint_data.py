import json
import os.path as osp
import time

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

from cogdl.data import Graph, Dataset
from cogdl.utils import download_url, accuracy, multilabel_f1, bce_with_logits_loss, cross_entropy_loss
from . import register_dataset
from .planetoid_data import index_to_mask


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
    if isinstance(item, list):
        labels = np.zeros((feats.shape[0], len(item)), dtype=float)
        for key, val in class_map.items():
            labels[int(key)] = np.array(val)
    else:
        labels = np.zeros(feats.shape[0], dtype=np.long)
        for key, val in class_map.items():
            labels[int(key)] = val

    labels = torch.from_numpy(labels)

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

    data = Graph(
        x=feats,
        y=labels,
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
        if len(self.data.y.shape) == 1:
            return int(torch.max(self.data.y) + 1)
        return self.data.y.shape[1]

    def download(self):
        for name in self.raw_file_names:
            download_url(self.url.format(name), self.raw_dir, name=name)
            time.sleep(0.5)

    def process(self):
        data = read_saint_data(self.raw_dir)
        torch.save(data, self.processed_paths[0])

    def get(self, idx):
        return self.data

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss

    def __repr__(self):
        return "{}()".format(self.name)

    def __len__(self):
        return self.data.x.shape[0]


def scale_feats(data):
    scaler = StandardScaler()
    feats = data.x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    data.x = feats
    return data


# def scale_feats(data):
#     x_sum = torch.sum(data.x, dim=1)
#     x_rev = x_sum.pow(-1).flatten()
#     x_rev[torch.isnan(x_rev)] = 0.0
#     x_rev[torch.isinf(x_rev)] = 0.0
#     data.x = data.x * x_rev.unsqueeze(-1).expand_as(data.x)
#     return data


@register_dataset("yelp")
class YelpDataset(SAINTDataset):
    def __init__(self, data_path="data"):
        dataset = "Yelp"
        url = "https://cloud.tsinghua.edu.cn/d/7218cc013c9a40159306/files/?p=%2F{}&dl=1"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(YelpDataset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss


@register_dataset("amazon-s")
class AmazonDataset(SAINTDataset):
    def __init__(self, data_path="data"):
        dataset = "AmazonSaint"
        url = "https://cloud.tsinghua.edu.cn/d/ae4b2c4f59bd41be9b0b/files/?p=%2F{}&dl=1"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(AmazonDataset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss


@register_dataset("flickr")
class FlickrDatset(SAINTDataset):
    def __init__(self, data_path="data"):
        dataset = "Flickr"
        url = "https://cloud.tsinghua.edu.cn/d/d3ebcb5fa2da463b8213/files/?p=%2F{}&dl=1"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(FlickrDatset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss


@register_dataset("reddit")
class RedditDataset(SAINTDataset):
    def __init__(self, data_path="data"):
        dataset = "Reddit"
        url = "https://cloud.tsinghua.edu.cn/d/d087e7e766e747ce8073/files/?p=%2F{}&dl=1"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(RedditDataset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)

    def get_evaluator(self):
        return accuracy

    def get_loss_fn(self):
        return cross_entropy_loss


@register_dataset("ppi")
class PPIDataset(SAINTDataset):
    def __init__(self, data_path="data"):
        dataset = "PPI"
        url = "https://cloud.tsinghua.edu.cn/d/2c6e94ec9dad4972b58e/files/?p=%2F{}&dl=1"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(PPIDataset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss


@register_dataset("ppi-large")
class PPILargeDataset(SAINTDataset):
    def __init__(self, data_path="data"):
        dataset = "PPI_Large"
        url = "https://cloud.tsinghua.edu.cn/d/469ef38a520640bba267/files/?p=%2F{}&dl=1"
        if data_path is None:
            data_path = "data"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            SAINTDataset(path, dataset, url)
        super(PPILargeDataset, self).__init__(path, dataset, url)
        self.data = scale_feats(self.data)

    def get_evaluator(self):
        return multilabel_f1

    def get_loss_fn(self):
        return bce_with_logits_loss
