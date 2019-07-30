from itertools import product
import os
import os.path as osp
import json

import torch
import numpy as np
import scipy
import networkx as nx
from networkx.readwrite import json_graph
from cogdl.data import (InMemoryDataset, Dataset, Data, download_url, extract_zip)
from cogdl.read import read_edgelist_label_data
import cogdl.transforms as T


from . import register_dataset

class EdgelistLabel(Dataset):
    r"""networks from the https://github.com/THUDM/ProNE/raw/master/data

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wikipedia"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`cogdl.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`cogdl.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/THUDM/ProNE/raw/master/data'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(EdgelistLabel, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        splits = [self.name]
        files = ['ungraph', 'cmty']
        return ['{}.{}'.format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_edgelist_label_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, self.processed_paths[0])
        

@register_dataset('dblp')
class DBLP(EdgelistLabel):
    def __init__(self):
        dataset = 'dblp'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        super(DBLP, self).__init__(path, dataset)
