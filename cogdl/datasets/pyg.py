import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit

from . import register_dataset


@register_dataset("cora")
class CoraDataset(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(CoraDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("citeseer")
class CiteSeerDataset(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(CiteSeerDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("pubmed")
class PubMedDataset(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(PubMedDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("reddit")
class RedditDataset(Reddit):
    def __init__(self):
        dataset = "Reddit"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(RedditDataset, self).__init__(path, T.TargetIndegree())
