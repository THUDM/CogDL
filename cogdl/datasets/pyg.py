import os.path as osp
import os
import torch
import numpy as np
import scipy.sparse as sp

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, TUDataset, QM9
from torch_geometric.utils import remove_self_loops
from . import register_dataset


def normalize_feature(data):
    x_sum = torch.sum(data.x, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[torch.isnan(x_rev)] = 0.
    x_rev[torch.isinf(x_rev)] = 0.
    data.x = data.x * x_rev.unsqueeze(-1).expand_as(data.x)
    return data


@register_dataset("cora")
class CoraDataset(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CoraDataset, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = normalize_feature(self.data)


@register_dataset("citeseer")
class CiteSeerDataset(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CiteSeerDataset, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = normalize_feature(self.data)


@register_dataset("pubmed")
class PubMedDataset(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(PubMedDataset, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = normalize_feature(self.data)


@register_dataset("reddit")
class RedditDataset(Reddit):
    def __init__(self):
        self.url = "https://data.dgl.ai/dataset/reddit.zip"
        dataset = "Reddit"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Reddit(path)
        super(RedditDataset, self).__init__(path, transform=T.TargetIndegree())


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


@register_dataset("reddit-multi-5k")
class RedditMulti5K(TUDataset):
    def __init__(self):
        dataset = "REDDIT-MULTI-5K"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditMulti5K, self).__init__(path, name=dataset)


@register_dataset("reddit-multi-12k")
class RedditMulti12K(TUDataset):
    def __init__(self):
        dataset = "REDDIT-MULTI-12K"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditMulti12K, self).__init__(path, name=dataset)


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


@register_dataset("enzymes")
class ENZYMES(TUDataset):
    def __init__(self):
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


@register_dataset("qm9")
class QM9Dataset(QM9):
    def __init__(self):
        dataset = "QM9"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)

        target = 0

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
            QM9(path)
        super(QM9Dataset, self).__init__(path)


def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx


def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_l1_normalize(X):
    norm = 1e-6 + X.sum(dim=1, keepdim=True)
    return X/norm


def preprocess_data_sgcpn(data, normalize_feature=True, missing_rate=0):
    data.train_mask = data.train_mask.type(torch.bool)
    data.val_mask = data.val_mask.type(torch.bool)
    # expand test_mask to all rest nodes
    data.test_mask = ~(data.train_mask + data.val_mask)
    # get adjacency matrix
    n = len(data.x)
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    adj = normalize_adj_row(adj)
    data.adj = to_torch_sparse(adj).to_dense()
    if normalize_feature:
        data.x = row_l1_normalize(data.x)
    erasing_pool = torch.arange(n)[~data.train_mask]
    size = int(len(erasing_pool) * (missing_rate / 100))
    idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
    if missing_rate > 0:
        data.x[idx_erased] = 0
    return data


@register_dataset("cora-missing-0")
class CoraDatasetMissing0(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CoraDatasetMissing0, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=0)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("citeseer-missing-0")
class CiteSeerDatasetMissing0(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CiteSeerDatasetMissing0, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=0)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("pubmed-missing-0")
class PubMedDatasetMissing0(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(PubMedDatasetMissing0, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=0)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice

@register_dataset("cora-missing-20")
class CoraDatasetMissing20(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CoraDatasetMissing20, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=20)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("citeseer-missing-20")
class CiteSeerDatasetMissing20(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CiteSeerDatasetMissing20, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=20)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("pubmed-missing-20")
class PubMedDatasetMissing20(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(PubMedDatasetMissing20, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=20)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice

@register_dataset("cora-missing-40")
class CoraDatasetMissing40(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CoraDatasetMissing40, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=40)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("citeseer-missing-40")
class CiteSeerDatasetMissing40(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CiteSeerDatasetMissing40, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=40)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("pubmed-missing-40")
class PubMedDatasetMissing40(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(PubMedDatasetMissing40, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=40)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice

@register_dataset("cora-missing-60")
class CoraDatasetMissing60(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CoraDatasetMissing60, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=60)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("citeseer-missing-60")
class CiteSeerDatasetMissing60(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CiteSeerDatasetMissing60, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=60)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("pubmed-missing-60")
class PubMedDatasetMissing60(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(PubMedDatasetMissing60, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=60)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice

@register_dataset("cora-missing-80")
class CoraDatasetMissing80(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CoraDatasetMissing80, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=80)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("citeseer-missing-80")
class CiteSeerDatasetMissing80(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CiteSeerDatasetMissing80, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=80)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("pubmed-missing-80")
class PubMedDatasetMissing80(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(PubMedDatasetMissing80, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=80)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice

@register_dataset("cora-missing-100")
class CoraDatasetMissing100(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CoraDatasetMissing100, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=100)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("citeseer-missing-100")
class CiteSeerDatasetMissing100(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(CiteSeerDatasetMissing100, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=100)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice


@register_dataset("pubmed-missing-100")
class PubMedDatasetMissing100(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        if not osp.exists(path):
            Planetoid(path, dataset, transform=T.TargetIndegree())
        super(PubMedDatasetMissing100, self).__init__(path, dataset, transform=T.TargetIndegree())
        self.data = preprocess_data_sgcpn(self.data, normalize_feature=True, missing_rate=100)
        adj_slice = torch.tensor(self.data.adj.size())
        adj_slice[0] = 0
        self.slices['adj'] = adj_slice