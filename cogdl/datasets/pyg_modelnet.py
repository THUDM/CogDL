import os
import os.path as osp
import shutil
import glob
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from cogdl.data import Data, Dataset, download_url
from . import register_dataset


class ModelNet10(ModelNet):
    def __init__(self, train):
        dataset = "ModelNet10"
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            ModelNet(path, "10", transform, pre_transform)
        super(ModelNet10, self).__init__(path, name="10", train=train, transform=transform, pre_transform=pre_transform)


class ModelNet40(ModelNet):
    def __init__(self, train):
        dataset = "ModelNet40"
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            ModelNet(path, "40", transform, pre_transform)
        super(ModelNet40, self).__init__(path, name="40", train=train, transform=transform, pre_transform=pre_transform)


@register_dataset("ModelNet10")
class ModelNetData10(ModelNet):
    def __init__(self):
        dataset = "ModelNet10"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        self.train_data = ModelNet10(True)
        self.test_data = ModelNet10(False)
        self.num_graphs = len(self.train_data) + len(self.test_data)

        super(ModelNetData10, self).__init__(path, name="10")

    def get_all(self):
        return self.train_data, self.test_data

    def __getitem__(self, item):
        if item < len(self.train_data):
            return self.train_data[item]
        return self.test_data[item]

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    @property
    def train_index(self):
        return 0, len(self.train_data)

    @property
    def test_index(self):
        return len(self.train_data), self.num_graphs


@register_dataset("ModelNet40")
class ModelNetData40(ModelNet):
    def __init__(self):
        dataset = "ModelNet40"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        self.train_data = ModelNet40(True)
        self.test_data = ModelNet40(False)
        self.num_graphs = len(self.train_data) + len(self.test_data)

        super(ModelNetData40, self).__init__(path, name="40")

    def get_all(self):
        return self.train_data, self.test_data

    def __getitem__(self, item):
        if item < len(self.train_data):
            return self.train_data[item]
        return self.test_data[item]

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    @property
    def train_index(self):
        return 0, len(self.train_data)

    @property
    def test_index(self):
        return len(self.train_data), self.num_graphs
