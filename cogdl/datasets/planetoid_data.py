import os.path as osp
import pickle as pkl
import sys

import numpy as np
import torch

from cogdl.data import Dataset, Graph
from cogdl.utils import remove_self_loops, download_url, untar, coalesce, Accuracy, CrossEntropyLoss


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def index_to_mask(index, size):
    mask = torch.full((size,), False, dtype=torch.bool)
    mask[index] = True
    return mask


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row.append(np.repeat(key, len(value)))
        col.append(np.array(value))
    _row = np.concatenate(row)
    _col = np.concatenate(col)
    edge_index = np.stack([_row, _col], axis=0)

    row_dom = edge_index[:, _row > _col]
    col_dom = edge_index[:, _col > _row][[1, 0]]
    edge_index = np.concatenate([row_dom, col_dom], axis=1)
    _row, _col = edge_index

    edge_index = np.stack([_row, _col], axis=0)

    order = np.lexsort((_col, _row))
    edge_index = edge_index[:, order]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # There may be duplicated edges and self loops in the datasets.
    edge_index, _ = remove_self_loops(edge_index)
    row = torch.cat([edge_index[0], edge_index[1]])
    col = torch.cat([edge_index[1], edge_index[0]])

    row, col, _ = coalesce(row, col)
    edge_index = torch.stack([row, col])
    return edge_index


def read_planetoid_data(folder, prefix):
    prefix = prefix.lower()
    names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    objects = []
    for item in names[:-1]:
        with open(f"{folder}/ind.{prefix}.{item}", "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))
    test_index = parse_index_file(f"{folder}/ind.{prefix}.{names[-1]}")
    test_index = torch.Tensor(test_index).long()
    test_index_reorder = test_index.sort()[0]

    x, tx, allx, y, ty, ally, graph = tuple(objects)
    x, tx, allx = tuple([torch.from_numpy(item.todense()).float() for item in [x, tx, allx]])
    y, ty, ally = tuple([torch.from_numpy(item).float() for item in [y, ty, ally]])

    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)

    if prefix.lower() == "citeseer":
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = torch.zeros(len_test_indices, tx.size(1))
        tx_ext[test_index_reorder - test_index.min(), :] = tx
        ty_ext = torch.zeros(len_test_indices, ty.size(1))
        ty_ext[test_index_reorder - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    x = torch.cat([allx, tx], dim=0).float()
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1].long()

    x[test_index] = x[test_index_reorder]
    y[test_index] = y[test_index_reorder]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Graph(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


class Planetoid(Dataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    """

    url = "https://cloud.tsinghua.edu.cn/d/6808093f7f8042bfa1f0/files/?p=%2F"

    def __init__(self, root, name, split="public", num_train_per_class=20, num_val=500, num_test=1000):
        self.name = name

        super(Planetoid, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

        self.split = split

        if split == "full":
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data = data

    @property
    def raw_file_names(self):
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return ["ind.{}.{}".format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def num_classes(self):
        assert hasattr(self.data, "y")
        return int(torch.max(self.data.y)) + 1

    @property
    def num_nodes(self):
        assert hasattr(self.data, "y")
        return self.data.y.shape[0]

    def download(self):
        fname = "{}.zip".format(self.name.lower())
        download_url("{}{}.zip&dl=1".format(self.url, self.name.lower()), self.raw_dir, fname)
        untar(self.raw_dir, fname)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        torch.save(data, self.processed_paths[0])

    def get(self, idx):
        return self.data

    def __repr__(self):
        return "{}".format(self.name)

    def __len__(self):
        return 1

    def get_evaluator(self):
        return Accuracy()

    def get_loss_fn(self):
        return CrossEntropyLoss()


def normalize_feature(data):
    x_sum = torch.sum(data.x, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[torch.isnan(x_rev)] = 0.0
    x_rev[torch.isinf(x_rev)] = 0.0
    data.x = data.x * x_rev.unsqueeze(-1).expand_as(data.x)
    return data


class CoraDataset(Planetoid):
    def __init__(self, data_path="data"):
        dataset = "Cora"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            Planetoid(path, dataset)
        super(CoraDataset, self).__init__(path, dataset)
        self.data = normalize_feature(self.data)


class CiteSeerDataset(Planetoid):
    def __init__(self, data_path="data"):
        dataset = "CiteSeer"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            Planetoid(path, dataset)
        super(CiteSeerDataset, self).__init__(path, dataset)
        self.data = normalize_feature(self.data)


class PubMedDataset(Planetoid):
    def __init__(self, data_path="data"):
        dataset = "PubMed"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            Planetoid(path, dataset)
        super(PubMedDataset, self).__init__(path, dataset)
        self.data = normalize_feature(self.data)
