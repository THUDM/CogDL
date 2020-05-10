import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, TUDataset

from . import register_dataset


@register_dataset("cora")
class CoraDataset(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, T.NormalizeFeatures())
        super(CoraDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("citeseer")
class CiteSeerDataset(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, T.NormalizeFeatures())
        super(CiteSeerDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("pubmed")
class PubMedDataset(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, T.NormalizeFeatures())
        super(PubMedDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("reddit")
class RedditDataset(Reddit):
    def __init__(self):
        dataset = "Reddit"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Reddit(path)
        super(RedditDataset, self).__init__(path, T.TargetIndegree())


@register_dataset("mutag")
class MUTAGDataset(TUDataset):
    def __init__(self):
        dataset = "MUTAG"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(MUTAGDataset, self).__init__(path, name=dataset)


@register_dataset("imdb-b")
class ImdbBinaryDataset(TUDataset):
    def __init__(self):
        dataset = "IMDB-BINARY"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ImdbBinaryDataset, self).__init__(path, name=dataset)


@register_dataset("imdb-m")
class ImdbMultiDataset(TUDataset):
    def __init__(self):
        dataset = "IMDB-MULTI"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ImdbMultiDataset, self).__init__(path, name=dataset)


@register_dataset("collab")
class CollabDataset(TUDataset):
    def __init__(self):
        dataset = "COLLAB"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(CollabDataset, self).__init__(path, name=dataset)


@register_dataset("proteins")
class ProtainsDataset(TUDataset):
    def __init__(self):
        dataset = "PROTEINS"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ProtainsDataset, self).__init__(path, name=dataset)