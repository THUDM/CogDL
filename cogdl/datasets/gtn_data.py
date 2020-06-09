import sys
import time
import os
import os.path as osp
import requests
import shutil
import tqdm
import pickle
import numpy as np

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
        self.url = f'https://github.com/cenyk1230/gtn-data/blob/master/{name}.zip?raw=true'
        super(GTNDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])
        self.num_classes = torch.max(self.data.train_target).item() + 1
        self.num_edge = len(self.data.adj)
        self.num_nodes = self.data.x.shape[0]


    @property
    def raw_file_names(self):
        names = ["edges.pkl", "labels.pkl", "node_features.pkl"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def read_gtn_data(self, folder):
        edges = pickle.load(open(osp.join(folder, 'edges.pkl'), 'rb'))
        labels = pickle.load(open(osp.join(folder, 'labels.pkl'), 'rb'))
        node_features = pickle.load(open(osp.join(folder, 'node_features.pkl'), 'rb'))

        data = Data()
        data.x = torch.from_numpy(node_features).type(torch.FloatTensor)

        num_nodes = edges[0].shape[0]

        node_type = np.zeros((num_nodes), dtype=int)
        assert len(edges)==4
        assert len(edges[0].nonzero())==2
        
        node_type[edges[0].nonzero()[0]] = 0
        node_type[edges[0].nonzero()[1]] = 1
        node_type[edges[1].nonzero()[0]] = 1
        node_type[edges[1].nonzero()[1]] = 0       
        node_type[edges[2].nonzero()[0]] = 0
        node_type[edges[2].nonzero()[1]] = 2
        node_type[edges[3].nonzero()[0]] = 2
        node_type[edges[3].nonzero()[1]] = 0
          
        print(node_type)
        data.pos = torch.from_numpy(node_type)
        
        edge_list = []
        for i, edge in enumerate(edges):
            edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.LongTensor)
            edge_list.append(edge_tmp)
        data.edge_index = torch.cat(edge_list, 1)
        
        A = []                     
        for i,edge in enumerate(edges):
            edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.LongTensor)
            value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
            A.append((edge_tmp,value_tmp))
        edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
        A.append((edge_tmp,value_tmp))
        data.adj = A

        data.train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.LongTensor)
        data.train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.LongTensor)
        data.valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.LongTensor)
        data.valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.LongTensor)
        data.test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.LongTensor)
        data.test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.LongTensor)
        
        y = np.zeros((num_nodes), dtype=int)
        x_index = torch.cat((data.train_node, data.valid_node, data.test_node))
        y_index = torch.cat((data.train_target, data.valid_target, data.test_target))
        y[x_index.numpy()] = y_index.numpy()
        data.y = torch.from_numpy(y)
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


@register_dataset("gtn-acm")
class ACM_GTNDataset(GTNDataset):
    def __init__(self):
        dataset = "gtn-acm"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(ACM_GTNDataset, self).__init__(path, dataset)


@register_dataset("gtn-dblp")
class DBLP_GTNDataset(GTNDataset):
    def __init__(self):
        dataset = "gtn-dblp"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(DBLP_GTNDataset, self).__init__(path, dataset)


@register_dataset("gtn-imdb")
class IMDB_GTNDataset(GTNDataset):
    def __init__(self):
        dataset = "gtn-imdb"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(IMDB_GTNDataset, self).__init__(path, dataset)
