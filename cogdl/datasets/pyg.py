import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, TUDataset, QM9
from torch_geometric.utils import remove_self_loops
from . import register_dataset


@register_dataset("cora")
class CoraDataset(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, T.NormalizeFeatures())
            Planetoid(path, dataset, T.TargetIndegree())
        super(CoraDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("citeseer")
class CiteSeerDataset(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, T.TargetIndegree())
        super(CiteSeerDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("pubmed")
class PubMedDataset(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, T.TargetIndegree())
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


@register_dataset("reddit-b")
class RedditBinary(TUDataset):
    def __init__(self):
        dataset = "REDDIT-BINARY"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditBinary, self).__init__(path, name=dataset)


@register_dataset("ptc-mr")
class PTCMRDataset(TUDataset):
    def __init__(self):
        dataset = "PTC_MR"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(PTCMRDataset, self).__init__(path, name=dataset)


@register_dataset("nci1")
class NCT1Dataset(TUDataset):
    def __init__(self):
        dataset = "NCI1"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(NCT1Dataset, self).__init__(path, name=dataset)


@register_dataset("nci109")
class NCT109Dataset(TUDataset):
    def __init__(self):
        dataset = "NCI109"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(NCT109Dataset, self).__init__(path, name=dataset)

@register_dataset("qm9")
class QM9Dataset(QM9):
    def __init__(self):
        dataset = "QM9"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)

        target=0

        class MyTransform(object):
            def __call__(self, data):
                # Specify target.
                data.y = data.y[:, target]
                return data

        class Complete(object):
            def __call__(self, data):
                device = data.edge_index.device
                row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
                col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
                row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
                col = col.repeat(data.num_nodes)
                edge_index = torch.stack([row, col], dim=0)
                edge_attr = None
                if data.edge_attr is not None:
                    idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
                    size = list(data.edge_attr.size())
                    size[0] = data.num_nodes * data.num_nodes
                    edge_attr = data.edge_attr.new_zeros(size)
                    edge_attr[idx] = data.edge_attr
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                data.edge_attr = edge_attr
                data.edge_index = edge_index
                return data

        transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
        if not osp.exists(path):
            QM9(path, transform=transform)
        super(QM9Dataset, self).__init__(path)