import sys
import time
import os
import os.path as osp
import requests
import shutil
import tqdm
import pickle
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

import torch

from cogdl.data import Data, Dataset, download_url

from . import register_dataset


def untar(path, fname, deleteTar=True):
    """
    Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print('unpacking ' + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

class HANDataset(Dataset):
    r"""The network datasets "ACM", "DBLP" and "IMDB" from the
    `"Heterogeneous Graph Attention Network"
    <https://arxiv.org/abs/1903.07293>`_ paper.
    
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"han-acm"`,
            :obj:`"han-dblp"`, :obj:`"han-imdb"`).
    """

    def __init__(self, root, name):
        self.name = name
        self.url = f'https://github.com/cenyk1230/han-data/blob/master/{name}.zip?raw=true'
        super(HANDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])
        self.num_classes = torch.max(self.data.train_target).item() + 1
        self.num_edge = len(self.data.adj)
        self.num_nodes = self.data.x.shape[0]

    @property
    def raw_file_names(self):
        names = ["data.mat"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def read_gtn_data(self, folder):
        data = sio.loadmat(osp.join(folder, 'data.mat'))
        if self.name == 'han-acm' or self.name == 'han-imdb':
            truelabels, truefeatures = data['label'], data['feature'].astype(float)
        elif self.name == 'han-dblp':
            truelabels, truefeatures = data['label'], data['features'].astype(float)
        num_nodes = truefeatures.shape[0]
        if self.name == 'han-acm':
            rownetworks = [data['PAP'] - np.eye(num_nodes), data['PLP'] - np.eye(num_nodes)]
        elif self.name == 'han-dblp':
            rownetworks = [data['net_APA'] - np.eye(num_nodes), data['net_APCPA'] - np.eye(num_nodes), data['net_APTPA'] - np.eye(num_nodes)]
        elif self.name == 'han-imdb':
            rownetworks = [data['MAM'] - np.eye(num_nodes), data['MDM'] - np.eye(num_nodes), data['MYM'] - np.eye(num_nodes)]

        y = truelabels
        train_idx = data['train_idx']
        val_idx = data['val_idx']
        test_idx = data['test_idx']

        train_mask = sample_mask(train_idx, y.shape[0])
        val_mask = sample_mask(val_idx, y.shape[0])
        test_mask = sample_mask(test_idx, y.shape[0])

        y_train = np.argmax(y[train_mask, :], axis=1)
        y_val = np.argmax(y[val_mask, :], axis=1)
        y_test = np.argmax(y[test_mask, :], axis=1)

        data = Data()
        A = []                     
        for i, edge in enumerate(rownetworks):
            edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.LongTensor)
            value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
            A.append((edge_tmp, value_tmp))
        edge_tmp = torch.stack((torch.arange(0,num_nodes), torch.arange(0,num_nodes))).type(torch.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
        A.append((edge_tmp, value_tmp))
        data.adj = A

        data.x = torch.from_numpy(truefeatures).type(torch.FloatTensor)

        data.train_node = torch.from_numpy(train_idx[0]).type(torch.LongTensor)
        data.train_target = torch.from_numpy(y_train).type(torch.LongTensor)
        data.valid_node = torch.from_numpy(val_idx[0]).type(torch.LongTensor)
        data.valid_target = torch.from_numpy(y_val).type(torch.LongTensor)
        data.test_node = torch.from_numpy(test_idx[0]).type(torch.LongTensor)
        data.test_target = torch.from_numpy(y_test).type(torch.LongTensor)

        self.data = data

    def get(self, idx):
        assert idx == 0
        return self.data
    
    def apply_to_device(self, device):
        self.data.x = self.data.x.to(device)

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
        download_url(self.url, self.raw_dir, name=self.name + '.zip')
        untar(self.raw_dir, self.name + '.zip')

    def process(self):
        self.read_gtn_data(self.raw_dir)
        torch.save(self.data, self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)


@register_dataset("han-acm")
class ACM_HANDataset(HANDataset):
    def __init__(self):
        dataset = "han-acm"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(ACM_HANDataset, self).__init__(path, dataset)


@register_dataset("han-dblp")
class DBLP_HANDataset(HANDataset):
    def __init__(self):
        dataset = "han-dblp"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(DBLP_HANDataset, self).__init__(path, dataset)


@register_dataset("han-imdb")
class IMDB_HANDataset(HANDataset):
    def __init__(self):
        dataset = "han-imdb"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(IMDB_HANDataset, self).__init__(path, dataset)
