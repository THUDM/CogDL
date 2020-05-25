import os
import os.path as osp
import shutil
import glob

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from cogdl.data import Data, Dataset, download_url, InMemoryDataset
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
        super(ModelNetData10, self).__init__(path, name="10")
        self.train_data = ModelNet10(True)
        self.test_data = ModelNet10(False)

    def get_all(self):
        return self.train_data, self.test_data


@register_dataset("ModelNet40")
class ModelNetData40(ModelNet):
    def __init__(self):
        dataset = "ModelNet40"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(ModelNetData40, self).__init__(path, name="40")
        self.train_data = ModelNet40(True)
        self.test_data = ModelNet40(False)

    def get_all(self):
        return self.train_data, self.test_data
