import os.path as osp
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch

from cogdl.data import Dataset, Graph
from cogdl.utils import download_url, untar


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]


def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for line in lines:
        tmps = line.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def statistics(dataset, train_data, valid_data, test_data):
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset in ["ali", "amazon-rec"]:
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users

    train_user_set = defaultdict(list)
    test_user_set = defaultdict(list)
    valid_user_set = defaultdict(list)

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))

    return n_users, n_items, train_user_set, valid_user_set, test_user_set


def build_sparse_graph(data_cf, n_users, n_items):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.0] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users + n_items, n_users + n_items))
    return _bi_norm_lap(mat)


def build_recommendation_data(dataset, train_cf, valid_cf, test_cf):
    n_users, n_items, train_user_set, valid_user_set, test_user_set = statistics(dataset, train_cf, valid_cf, test_cf)

    print("building the adj mat ...")
    norm_mat = build_sparse_graph(train_cf, n_users, n_items)

    n_params = {
        "n_users": int(n_users),
        "n_items": int(n_items),
    }
    user_dict = {
        "train_user_set": train_user_set,
        "valid_user_set": valid_user_set if dataset != "yelp2018" else None,
        "test_user_set": test_user_set,
    }

    print("loading done.")
    data = Graph()
    data.train_cf = train_cf
    data.user_dict = user_dict
    data.n_params = n_params
    data.norm_mat = norm_mat

    return data


def read_recommendation_data(data_path, dataset):
    directory = data_path + "/"

    if dataset == "yelp2018":
        read_cf = read_cf_yelp2018
    else:
        read_cf = read_cf_amazon

    print("reading train and test user-item set ...")
    train_cf = read_cf(directory + "train.txt")
    test_cf = read_cf(directory + "test.txt")
    if dataset != "yelp2018":
        valid_cf = read_cf(directory + "valid.txt")
    else:
        valid_cf = test_cf
    data = build_recommendation_data(dataset, train_cf, valid_cf, test_cf)

    return data


class RecDataset(Dataset):
    r"""The recommendation datasets "Amazon", "Yelp2018" and "Ali" from the
    `"MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems"
    <https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf>`_ paper.
    """

    url = "https://cloud.tsinghua.edu.cn/d/ddbbff157971449eb163/files/?p=%2F"

    def __init__(self, root, name):
        self.name = name

        super(RecDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

        self.raw_dir = osp.join(self.root, self.name, "raw")
        self.processed_dir = osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        names = ["train.txt", "valid.txt", "test.txt"]
        return names

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        fname = "{}.zip".format(self.name.lower())
        download_url("{}{}.zip&dl=1".format(self.url, self.name.lower()), self.raw_dir, fname)
        untar(self.raw_dir, fname)

    def process(self):
        data = read_recommendation_data(self.raw_dir, self.name)
        torch.save(data, self.processed_paths[0])

    def get(self, idx):
        return self.data

    def __repr__(self):
        return "{}()".format(self.name)


class Yelp2018Dataset(RecDataset):
    def __init__(self, data_path="data"):
        dataset = "yelp2018"
        path = osp.join(data_path, dataset)
        super(Yelp2018Dataset, self).__init__(path, dataset)


class AliDataset(RecDataset):
    def __init__(self, data_path="data"):
        dataset = "ali"
        path = osp.join(data_path, dataset)
        super(AliDataset, self).__init__(path, dataset)


class AmazonRecDataset(RecDataset):
    def __init__(self, data_path="data"):
        dataset = "amazon-rec"
        path = osp.join(data_path, dataset)
        super(AmazonRecDataset, self).__init__(path, dataset)
