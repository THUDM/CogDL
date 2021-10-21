import os.path as osp

import torch

from cogdl.data import Graph, Dataset
from cogdl.utils import download_url


def read_gatne_data(folder):
    train_data = {}
    with open(osp.join(folder, "{}".format("train.txt")), "r") as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in train_data:
                train_data[items[0]] = []
            train_data[items[0]].append([int(items[1]), int(items[2])])

    valid_data = {}
    with open(osp.join(folder, "{}".format("valid.txt")), "r") as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in valid_data:
                valid_data[items[0]] = [[], []]
            valid_data[items[0]][1 - int(items[3])].append([int(items[1]), int(items[2])])

    test_data = {}
    with open(osp.join(folder, "{}".format("test.txt")), "r") as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in test_data:
                test_data[items[0]] = [[], []]
            test_data[items[0]][1 - int(items[3])].append([int(items[1]), int(items[2])])

    data = Graph()
    data.train_data = train_data
    data.valid_data = valid_data
    data.test_data = test_data
    return data


class GatneDataset(Dataset):
    r"""The network datasets "Amazon", "Twitter" and "YouTube" from the
    `"Representation Learning for Attributed Multiplex Heterogeneous Network"
    <https://arxiv.org/abs/1905.01669>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Amazon"`,
            :obj:`"Twitter"`, :obj:`"YouTube"`).
    """

    url = "https://github.com/THUDM/GATNE/raw/master/data"

    def __init__(self, root, name):
        self.name = name
        super(GatneDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ["train.txt", "valid.txt", "test.txt"]
        return names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}/{}".format(self.url, self.name.lower(), name), self.raw_dir)

    def process(self):
        data = read_gatne_data(self.raw_dir)
        torch.save(data, self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)


class AmazonDataset(GatneDataset):
    def __init__(self, data_path="data"):
        dataset = "amazon"
        path = osp.join(data_path, dataset)
        super(AmazonDataset, self).__init__(path, dataset)


class TwitterDataset(GatneDataset):
    def __init__(self, data_path="data"):
        dataset = "twitter"
        path = osp.join(data_path, dataset)
        super(TwitterDataset, self).__init__(path, dataset)


class YouTubeDataset(GatneDataset):
    def __init__(self, data_path="data"):
        dataset = "youtube"
        path = osp.join(data_path, dataset)
        super(YouTubeDataset, self).__init__(path, dataset)
