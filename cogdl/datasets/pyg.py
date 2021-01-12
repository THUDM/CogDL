import os.path as osp
import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, TUDataset, QM9
from torch_geometric.utils import remove_self_loops
from . import register_dataset
from cogdl.utils import accuracy_evaluator


def normalize_feature(data):
    x_sum = torch.sum(data.x, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[torch.isnan(x_rev)] = 0.0
    x_rev[torch.isinf(x_rev)] = 0.0
    data.x = data.x * x_rev.unsqueeze(-1).expand_as(data.x)
    return data


@register_dataset("reddit")
class RedditDataset(Reddit):
    def __init__(self, args=None):
        self.url = "https://data.dgl.ai/dataset/reddit.zip"
        dataset = "Reddit"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Reddit(path)
        super(RedditDataset, self).__init__(path, transform=T.TargetIndegree())

    def get_evaluator(self):
        return accuracy_evaluator()


@register_dataset("mutag")
class MUTAGDataset(TUDataset):
    def __init__(self, args=None):
        dataset = "MUTAG"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(MUTAGDataset, self).__init__(path, name=dataset)


@register_dataset("imdb-b")
class ImdbBinaryDataset(TUDataset):
    def __init__(self, args=None):
        dataset = "IMDB-BINARY"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ImdbBinaryDataset, self).__init__(path, name=dataset)


@register_dataset("imdb-m")
class ImdbMultiDataset(TUDataset):
    def __init__(self, args=None):
        dataset = "IMDB-MULTI"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ImdbMultiDataset, self).__init__(path, name=dataset)


@register_dataset("collab")
class CollabDataset(TUDataset):
    def __init__(self, args=None):
        dataset = "COLLAB"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(CollabDataset, self).__init__(path, name=dataset)


@register_dataset("proteins")
class ProtainsDataset(TUDataset):
    def __init__(self, args=None):
        dataset = "PROTEINS"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ProtainsDataset, self).__init__(path, name=dataset)


@register_dataset("reddit-b")
class RedditBinary(TUDataset):
    def __init__(self, args=None):
        dataset = "REDDIT-BINARY"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditBinary, self).__init__(path, name=dataset)


@register_dataset("reddit-multi-5k")
class RedditMulti5K(TUDataset):
    def __init__(self, args=None):
        dataset = "REDDIT-MULTI-5K"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditMulti5K, self).__init__(path, name=dataset)


@register_dataset("reddit-multi-12k")
class RedditMulti12K(TUDataset):
    def __init__(self, args=None):
        dataset = "REDDIT-MULTI-12K"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditMulti12K, self).__init__(path, name=dataset)


@register_dataset("ptc-mr")
class PTCMRDataset(TUDataset):
    def __init__(self, args=None):
        dataset = "PTC_MR"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(PTCMRDataset, self).__init__(path, name=dataset)


@register_dataset("nci1")
class NCT1Dataset(TUDataset):
    def __init__(self, args=None):
        dataset = "NCI1"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(NCT1Dataset, self).__init__(path, name=dataset)


@register_dataset("nci109")
class NCT109Dataset(TUDataset):
    def __init__(self, args=None):
        dataset = "NCI109"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(NCT109Dataset, self).__init__(path, name=dataset)


@register_dataset("enzymes")
class ENZYMES(TUDataset):
    def __init__(self, args=None):
        dataset = "ENZYMES"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ENZYMES, self).__init__(path, name=dataset)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            edge_nodes = data.edge_index.max() + 1
            if edge_nodes < data.x.size(0):
                data.x = data.x[:edge_nodes]
            return data
        else:
            return self.index_select(idx)
