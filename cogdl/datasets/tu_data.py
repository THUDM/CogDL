import glob
import os
import os.path as osp
import shutil
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from cogdl.data.dataset import MultiGraphDataset
from cogdl.data import Graph
from cogdl.utils import download_url


def normalize_feature(data):
    x_sum = torch.sum(data.x, dim=1)
    x_rev = x_sum.pow(-1).flatten()
    x_rev[torch.isnan(x_rev)] = 0.0
    x_rev[torch.isinf(x_rev)] = 0.0
    data.x = data.x * x_rev.unsqueeze(-1).expand_as(data.x)
    return data


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, "r") as f:
        src = f.read().split("\n")[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, "{}_{}.txt".format(prefix, name))
    return read_txt_array(path, sep=",", dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def _split(edge_index, batch, x=None, y=None, edge_attr=None):
    node_slice = np.bincount(batch).tolist()
    row, _ = edge_index
    edge_slice = np.bincount(batch[row]).tolist()
    if edge_attr is not None:
        edge_attr = edge_attr.split(edge_slice)
    edge_index_t = edge_index.T.split(edge_slice)
    if x is not None:
        x = x.split(node_slice)
        num_nodes = [i.shape[0] for i in x]
        num_nodes_cum = np.cumsum(num_nodes).tolist()
    else:
        num_nodes_cum = [edge.max().item() + 1 for edge in edge_index_t]

    num_nodes_cum = [0] + num_nodes_cum
    if edge_index_t[-1].min() > 0:
        edge_index_t = [edge_index_t[i] - num_nodes_cum[i] for i in range(len(edge_index_t))]
    data = []
    for i in range(len(node_slice)):
        g = Graph(edge_index=edge_index_t[i].T)
        if x is not None:
            g.x = x[i]
        if y is not None:
            g.y = y[i].view(1)
        if edge_attr is not None:
            g.edge_attr = edge_attr[i]
        data.append(g)
    return data


def segment(src, indptr):
    out_list = []
    for i in range(indptr.size(-1) - 1):
        indexptr = torch.arange(indptr[..., i].item(), indptr[..., i + 1].item(), dtype=torch.int64)
        src_data = src.index_select(indptr.dim() - 1, indexptr)
        out = torch.sum(src_data, dim=indptr.dim() - 1, keepdim=True)
        out_list.append(out)
    return torch.cat(out_list, dim=indptr.dim() - 1)


def coalesce(index, value, m, n):
    row = index[0]
    col = index[1]

    idx = col.new_zeros(col.numel() + 1)
    idx[1:] = row
    idx[1:] *= n
    idx[1:] += col
    if (idx[1:] < idx[:-1]).any():
        perm = idx[1:].argsort()
        row = row[perm]
        col = col[perm]
        if value is not None:
            value = value[perm]

    idx = col.new_full((col.numel() + 1,), -1)
    idx[1:] = n * row + col
    mask = idx[1:] > idx[:-1]

    if mask.all():  # Skip if indices are already coalesced.
        return torch.stack([row, col], dim=0), value

    row = row[mask]
    col = col[mask]

    if value is not None:
        ptr = mask.nonzero().flatten()
        ptr = torch.cat([ptr, ptr.new_full((1,), value.size(0))])
        value = segment(value, ptr)
        value = value[0] if isinstance(value, tuple) else value

    return torch.stack([row, col], dim=0), value


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, "{}_*.txt".format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1 : -4] for f in files]

    edge_index = read_file(folder, prefix, "A", torch.long).t() - 1
    batch = read_file(folder, prefix, "graph_indicator", torch.long) - 1

    node_attributes = node_labels = None
    if "node_attributes" in names:
        node_attributes = read_file(folder, prefix, "node_attributes")
    if "node_labels" in names:
        node_labels = read_file(folder, prefix, "node_labels", torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
    x = cat([node_attributes, node_labels])

    edge_attributes, edge_labels = None, None
    if "edge_attributes" in names:
        edge_attributes = read_file(folder, prefix, "edge_attributes")
    if "edge_labels" in names:
        edge_labels = read_file(folder, prefix, "edge_labels", torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])

    y = None
    if "graph_attributes" in names:  # Regression problem.
        y = read_file(folder, prefix, "graph_attributes")
    elif "graph_labels" in names:  # Classification problem.
        y = read_file(folder, prefix, "graph_labels", torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
    if x is not None:
        x = x[:, num_node_attributes(x) :]
    if edge_attr is not None:
        edge_attr = edge_attr[:, : num_edge_attributes(edge_attr)]

    graphs = _split(edge_index, batch=batch, x=x, y=y, edge_attr=edge_attr)
    return graphs, y


def num_node_labels(x=None):
    if x is None:
        return 0
    for i in range(x.size(1)):
        _x = x[:, i:]
        if ((_x == 0) | (_x == 1)).all() and (_x.sum(dim=1) == 1).all():
            return x.size(1) - i
    return 0


def num_node_attributes(x=None):
    if x is None:
        return 0
    return x.size(1) - num_node_labels(x)


def num_edge_labels(edge_attr=None):
    if edge_attr is None:
        return 0
    for i in range(edge_attr.size(1)):
        if edge_attr[:, i:].sum() == edge_attr.size(0):
            return edge_attr.size(1) - i
    return 0


def num_edge_attributes(edge_attr=None):
    if edge_attr is None:
        return 0
    return edge_attr.size(1) - num_edge_labels(edge_attr)


class TUDataset(MultiGraphDataset):
    url = "https://www.chrsmrrs.com/graphkerneldatasets"

    def __init__(self, root, name):
        self.name = name
        super(TUDataset, self).__init__(root)
        # self.data = torch.load(self.processed_paths[0])

        # if self.data[0].x is not None:
        #     num_node_attributes = self.num_node_attributes
        #     self.data.x = self.data.x[:, num_node_attributes:]
        # if self.data.edge_attr is not None:
        #     num_edge_attributes = self.num_edge_attributes
        #     self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

        self.data, self.y = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ["A", "graph_indicator"]
        return ["{}_{}.txt".format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        url = self.url
        folder = osp.join(self.root)
        path = download_url("{}/{}.zip".format(url, self.name), folder)
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        data = read_tu_data(self.raw_dir, self.name)
        torch.save(data, self.processed_paths[0])

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        return self.y.max().item() + 1 if self.y.dim() == 1 else self.y.size(1)

    def __len__(self):
        return len(self.data)


class MUTAGDataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "MUTAG"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(MUTAGDataset, self).__init__(path, name=dataset)


class ImdbBinaryDataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "IMDB-BINARY"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ImdbBinaryDataset, self).__init__(path, name=dataset)


class ImdbMultiDataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "IMDB-MULTI"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ImdbMultiDataset, self).__init__(path, name=dataset)


class CollabDataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "COLLAB"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(CollabDataset, self).__init__(path, name=dataset)


class ProteinsDataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "PROTEINS"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(ProteinsDataset, self).__init__(path, name=dataset)


class RedditBinary(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "REDDIT-BINARY"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditBinary, self).__init__(path, name=dataset)


class RedditMulti5K(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "REDDIT-MULTI-5K"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditMulti5K, self).__init__(path, name=dataset)


class RedditMulti12K(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "REDDIT-MULTI-12K"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(RedditMulti12K, self).__init__(path, name=dataset)


class PTCMRDataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "PTC_MR"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(PTCMRDataset, self).__init__(path, name=dataset)


class NCI1Dataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "NCI1"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(NCI1Dataset, self).__init__(path, name=dataset)


class NCI109Dataset(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "NCI109"
        path = osp.join(data_path, dataset)
        if not osp.exists(path):
            TUDataset(path, name=dataset)
        super(NCI109Dataset, self).__init__(path, name=dataset)


class ENZYMES(TUDataset):
    def __init__(self, data_path="data"):
        dataset = "ENZYMES"
        path = osp.join(data_path, dataset)
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
