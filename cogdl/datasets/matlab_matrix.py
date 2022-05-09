import os.path as osp
from itertools import product

import numpy as np
import scipy.io
import torch
import time

from cogdl.data import Graph, Dataset
from cogdl.utils import download_url


class MatlabMatrix(Dataset):
    r"""networks from the http://leitang.net/code/social-dimension/data/ or http://snap.stanford.edu/node2vec/

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Blogcatalog"`).
    """

    def __len__(self):
        return 1

    def __init__(self, root, name, url):
        self.name = name
        self.url = url
        super(MatlabMatrix, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        splits = [self.name]
        files = ["mat"]
        return ["{}.{}".format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def num_classes(self):
        return self.data.y.shape[1]

    @property
    def num_nodes(self):
        return self.data.y.shape[0]

    def download(self):
        for name in self.raw_file_names:
            download_url("{}{}".format(self.url, name), self.raw_dir)

    def get(self, idx):
        assert idx == 0
        return self.data

    def process(self):
        path = osp.join(self.raw_dir, "{}.mat".format(self.name))
        smat = scipy.io.loadmat(path)
        adj_matrix, group = smat["network"], smat["group"]

        y = torch.from_numpy(group.todense()).to(torch.float)

        row_ind, col_ind = adj_matrix.nonzero()
        edge_index = torch.stack([torch.tensor(row_ind), torch.tensor(col_ind)], dim=0)
        edge_attr = torch.tensor(adj_matrix[row_ind, col_ind])

        data = Graph(edge_index=edge_index, edge_attr=edge_attr, x=None, y=y)

        torch.save(data, self.processed_paths[0])


class BlogcatalogDataset(MatlabMatrix):
    def __init__(self, data_path="data"):
        dataset, filename = "blogcatalog", "blogcatalog"
        url = "http://leitang.net/code/social-dimension/data/"
        path = osp.join(data_path, dataset)
        super(BlogcatalogDataset, self).__init__(path, filename, url)


class FlickrDataset(MatlabMatrix):
    def __init__(self, data_path="data"):
        dataset, filename = "flickr", "flickr"
        url = "http://leitang.net/code/social-dimension/data/"
        path = osp.join(data_path, dataset)
        super(FlickrDataset, self).__init__(path, filename, url)


class WikipediaDataset(MatlabMatrix):
    def __init__(self, data_path="data"):
        dataset, filename = "wikipedia", "POS"
        url = "http://snap.stanford.edu/node2vec/"
        path = osp.join(data_path, dataset)
        super(WikipediaDataset, self).__init__(path, filename, url)


class PPIDataset(MatlabMatrix):
    def __init__(self, data_path="data"):
        dataset, filename = "ppi", "Homo_sapiens"
        url = "http://snap.stanford.edu/node2vec/"
        path = osp.join(data_path, dataset + "-ne")
        super(PPIDataset, self).__init__(path, filename, url)


class NetworkEmbeddingCMTYDataset(Dataset):
    def __init__(self, root, name, url):
        self.url = url
        self.name = name
        super(NetworkEmbeddingCMTYDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.name}.{x}" for x in ["ungraph", "cmty"]]

    @property
    def num_classes(self):
        return self.data.y.shape[1]

    @property
    def num_nodes(self):
        return self.data.y.shape[0]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url(self.url.format(name), self.raw_dir, name=name)
            time.sleep(0.5)

    def process(self):
        filenames = self.raw_paths
        with open(f"{filenames[0]}", "r") as f:
            edge_index = f.read().strip().split("\n")
        edge_index = [[int(i) for i in x.split("\t")] for x in edge_index]
        edge_index = np.array(edge_index, dtype=np.int64).transpose()
        edge_index = torch.from_numpy(edge_index)
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat((edge_index, rev_edge_index), dim=1)

        self_loop_mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, self_loop_mask]

        with open(f"{filenames[1]}", "r") as f:
            cmty = f.read().strip().split("\n")
        cmty = [[int(i) for i in x.split("\t")] for x in cmty]

        num_classes = len(cmty)
        num_nodes = torch.max(edge_index).item() + 1

        labels = np.zeros((num_nodes, num_classes), dtype=np.float)
        for i, cls in enumerate(cmty):
            labels[cls, i] = 1.0

        labels = torch.from_numpy(labels)
        data = Graph(x=None, y=labels, edge_index=edge_index)
        torch.save(data, self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)

    def __len__(self):
        return self.data.y.shape[0]


class DblpNEDataset(NetworkEmbeddingCMTYDataset):
    def __init__(self, data_path="data"):
        dataset = "dblp"
        path = osp.join(data_path, dataset + "-ne")
        url = "https://cloud.tsinghua.edu.cn/d/5ba8b35db80343549c67/files/?p=%2F{}&dl=1"
        super(DblpNEDataset, self).__init__(path, dataset, url)


class YoutubeNEDataset(NetworkEmbeddingCMTYDataset):
    def __init__(self, data_path="data"):
        dataset = "youtube"
        path = osp.join(data_path, dataset + "-ne")
        url = "https://cloud.tsinghua.edu.cn/d/c1ae63c4f1f14afb8ab8/files/?p=%2F{}&dl=1"
        super(YoutubeNEDataset, self).__init__(path, dataset, url)
