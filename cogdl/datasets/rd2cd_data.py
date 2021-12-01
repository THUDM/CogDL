import os
import random
import tarfile

import numpy as np
import requests
import torch
from torch import Tensor

from cogdl.data import Graph
from cogdl.datasets import NodeDataset

base_url = "https://cloud.tsinghua.edu.cn/d/65d7c53dd8474d7091a9/files/?p=%2F"


def get_whole_mask(y, ratio: str, seed: int = 1234567):
    """split the whole dataset in proportion"""
    y_have_label_mask = y != -1
    total_node_num = len(y)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    masked_index = y_index_tensor[y_have_label_mask]
    while True:
        (train_mask, val_mask, test_mask) = get_order(ratio, masked_index, total_node_num, seed)
        if check_train_containing(train_mask, y):
            return (train_mask, val_mask, test_mask)
        else:
            seed += 1


def get_order(ratio: str, masked_index: Tensor, total_node_num: int, seed: int = 1234567):
    """
    return：(train_mask,val_mask,test_mask)
    """
    random.seed(seed)

    masked_node_num = len(masked_index)
    shuffle_criterion = list(range(masked_node_num))
    random.shuffle(shuffle_criterion)

    train_val_test_list = [int(i) for i in ratio.split("-")]
    tvt_sum = sum(train_val_test_list)
    tvt_ratio_list = [i / tvt_sum for i in train_val_test_list]

    train_end_index = int(tvt_ratio_list[0] * masked_node_num)
    val_end_index = train_end_index + int(tvt_ratio_list[1] * masked_node_num)

    train_mask_index = shuffle_criterion[:train_end_index]
    val_mask_index = shuffle_criterion[train_end_index:val_end_index]
    test_mask_index = shuffle_criterion[val_end_index:]

    train_mask = torch.zeros(total_node_num, dtype=torch.bool)
    train_mask[masked_index[train_mask_index]] = True
    val_mask = torch.zeros(total_node_num, dtype=torch.bool)
    val_mask[masked_index[val_mask_index]] = True
    test_mask = torch.zeros(total_node_num, dtype=torch.bool)
    test_mask[masked_index[test_mask_index]] = True

    return (train_mask, val_mask, test_mask)


def check_train_containing(train_mask, y):
    for label in y.unique():
        if label.item() == -1:
            continue
        if label.item() not in y[train_mask]:
            return False
    return True


class RD2CD(NodeDataset):
    def __init__(self, root, name):
        self.root = root
        self.name = name
        self.source_path = root + "/" + name + "/raw"
        if not os.path.exists(self.source_path):
            os.makedirs(self.source_path)
        dst_path = root + "/" + name + "/processed"
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.data_path = dst_path + "/data.pt"
        if not os.path.exists(self.data_path):
            self.download()
        super(RD2CD, self).__init__(path=self.data_path, scale_feat=False)

    def download(self):
        r = requests.get(base_url + self.name + ".tgz&dl=1")
        tarfile_path = self.source_path + "/" + self.name + ".tgz"
        with open(tarfile_path, "wb") as f:
            f.write(r.content)
        with tarfile.open(tarfile_path, "r") as f:
            f.extractall(self.source_path)

    def process(self):
        numpy_x = np.load(self.source_path + "/x.npy")
        x = torch.from_numpy(numpy_x).to(torch.float)
        numpy_y = np.load(self.source_path + "/y.npy")
        y = torch.from_numpy(numpy_y).to(torch.long)
        numpy_edge_index = np.load(self.source_path + "/edge_index.npy")
        edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)

        # set train/val/test mask in node_classification task
        random_seed = 14530529  # a fixed seed
        (train_mask, val_mask, test_mask) = get_whole_mask(y, "6-2-2", random_seed)
        data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        torch.save(data, self.data_path)
        return data


class Github(RD2CD):
    def __init__(self, root="data"):
        super(Github, self).__init__(root, "Github")


class Elliptic(RD2CD):
    def __init__(self, root="data"):
        super(Elliptic, self).__init__(root, "Elliptic")


class Film(RD2CD):
    def __init__(self, root="data"):
        super(Film, self).__init__(root, "Film")


class Wiki(RD2CD):
    def __init__(self, root="data"):
        super(Wiki, self).__init__(root, "Wiki")


class Clothing(RD2CD):
    def __init__(self, root="data"):
        super(Clothing, self).__init__(root, "Clothing")


class Electronics(RD2CD):
    def __init__(self, root="data"):
        super(Electronics, self).__init__(root, "Electronics")


class Dblp(RD2CD):
    def __init__(self, root="data"):
        super(Dblp, self).__init__(root, "Dblp")


class Yelpchi(RD2CD):
    def __init__(self, root="data"):
        super(Yelpchi, self).__init__(root, "Yelpchi")


class Alpha(RD2CD):
    def __init__(self, root="data"):
        super(Alpha, self).__init__(root, "Alpha")


class Weibo(RD2CD):
    def __init__(self, root="data"):
        super(Weibo, self).__init__(root, "Weibo")


class bgp(RD2CD):
    def __init__(self, root="data"):
        super(bgp, self).__init__(root, "bgp")


class ssn5(RD2CD):
    def __init__(self, root="data"):
        super(ssn5, self).__init__(root, "ssn5")


class ssn7(RD2CD):
    def __init__(self, root="data"):
        super(ssn7, self).__init__(root, "ssn7")


class Aids(RD2CD):
    def __init__(self, root="data"):
        super(Aids, self).__init__(root, "Aids")


class Nba(RD2CD):
    def __init__(self, root="data"):
        super(Nba, self).__init__(root, "Nba")


class Pokec_z(RD2CD):
    def __init__(self, root="data"):
        super(Pokec_z, self).__init__(root, "Pokec_z")
