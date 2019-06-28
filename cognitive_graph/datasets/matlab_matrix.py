from itertools import product
import os
import os.path as osp
import json

import torch
import numpy as np
import scipy
import networkx as nx
from networkx.readwrite import json_graph
from cognitive_graph.data import (InMemoryDataset, Dataset, Data, download_url, extract_zip)
from cognitive_graph.utils import remove_self_loops
import cognitive_graph.transforms as T


from . import register_dataset

class MatlabMatrix(Dataset):
    r"""blogcatalog networks from the http://leitang.net/code/social-dimension/data/

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Blogcatalog"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`cognitive_graph.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`cognitive_graph.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'http://leitang.net/code/social-dimension/data/'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(MatlabMatrix, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        splits = [self.name]
        files = ['mat']
        return ['{}.{}'.format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def get(self, idx):
        assert idx == 0
        return self.data

    def process(self):
        path = osp.join(self.raw_dir, '{}.mat'.format(self.name.lower()))
        smat = scipy.io.loadmat(path)
        adj_matrix, group = smat["network"], smat["group"]

        y = torch.from_numpy(group.todense()).to(torch.float)

        row_ind, col_ind = adj_matrix.nonzero()
        edge_index = torch.stack([torch.tensor(row_ind), torch.tensor(col_ind)], dim=0)

        data = Data(edge_index=edge_index, x=None, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])


@register_dataset('blogcatalog')
class BlogcatalogDataset(MatlabMatrix):
    def __init__(self):
        dataset = 'blogcatalog'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        super(BlogcatalogDataset, self).__init__(path, dataset)


@register_dataset('flickr')
class FlickrDataset(MatlabMatrix):
    def __init__(self):
        dataset = 'flickr'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        super(FlickrDataset, self).__init__(path, dataset)
