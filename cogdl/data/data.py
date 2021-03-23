import re
import copy
from contextlib import contextmanager
import scipy.sparse as sp

import torch
import numpy as np
from cogdl.utils import (
    csr2coo,
    coo2csr_index,
    add_remaining_self_loops,
    remove_self_loops,
    symmetric_normalization,
    row_normalization,
    fast_spmm,
    get_degrees,
)
from cogdl.operators.operators import sample_adj_c, subgraph_c

indicator = fast_spmm is None


class BaseGraph(object):
    def __init__(self):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if self[key] is not None:
                yield key, self[key]

    def cat_dim(self, key, value):
        r"""Returns the dimension in which the attribute :obj:`key` with
        content :obj:`value` gets concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if bool(re.search("(index|face)", key)) else 0

    def __inc__(self, key, value):
        r""" "Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` should be cumulatively summed up when
        # creating batches.
        return self.num_nodes if bool(re.search("(index|face)", key)) else 0

    def __cat_dim__(self, key, value=None):
        return self.cat_dim(key, value)

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, :obj:`func` is applied to all present
        attributes.
        """
        for key, item in self(*keys):
            if isinstance(item, Adjacency):
                self[key] = func(item)
            if not isinstance(item, torch.Tensor):
                continue
            self[key] = func(item)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device), *keys)

    def cuda(self, *keys):
        return self.apply(lambda x: x.cuda(), *keys)


class Adjacency(BaseGraph):
    def __init__(self, row=None, col=None, row_ptr=None, weight=None, attr=None, num_nodes=None, **kwargs):
        super(Adjacency, self).__init__()
        self.row = row
        self.col = col
        self.row_ptr = row_ptr
        self.weight = weight
        self.attr = attr
        self.__num_nodes__ = num_nodes
        self.__normed__ = None
        self.__in_norm__ = self.__out_norm__ = None
        self.__symmetric__ = True
        for key, item in kwargs.items():
            self[key] = item

    def add_remaining_self_loops(self):
        edge_index = torch.stack([self.row, self.col])
        edge_index, self.weight = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)
        self.row, self.col = edge_index
        if indicator is True:
            self._to_csr()

    def remove_self_loops(self):
        edge_index = torch.stack([self.row, self.col])
        edge_index, self.weight = remove_self_loops(edge_index, self.weight)
        self.row, self.col = edge_index
        if indicator is True:
            self._to_csr()

    def sym_norm(self):
        if self.row is None:
            self.generate_normalization("sym")
        else:
            self.normalize_adj("sym")

    def row_norm(self):
        if self.row is None:
            self.generate_normalization("row")
        else:
            self.normalize_adj("row")
            self.__symmetric__ = False

    def generate_normalization(self, norm="sym"):
        if self.__normed__:
            return
        degrees = (self.row_ptr[1:] - self.row_ptr[:-1]).float()
        if norm == "sym":
            edge_norm = torch.pow(degrees, -0.5).to(self.device)
            edge_norm[torch.isinf(edge_norm)] = 0
            self.__out_norm__ = self.__in_norm__ = edge_norm.view(-1, 1)
        elif norm == "row":
            edge_norm = torch.pow(degrees, -1).to(self.device)
            edge_norm[torch.isinf(edge_norm)] = 0
            self.__out_norm__ = None
            self.__in_norm__ = edge_norm.view(-1, 1)
        else:
            raise NotImplementedError
        self.__normed__ = norm

    def normalize_adj(self, norm="sym"):
        if self.__normed__:
            return
        if self.weight is None or self.weight.shape[0] != self.edge_index.shape[1]:
            self.weight = torch.ones(self.num_edges, device=self.device)

        edge_index = torch.stack([self.row, self.col])
        if norm == "sym":
            self.weight = symmetric_normalization(self.num_nodes, edge_index, self.weight)
        elif norm == "row":
            self.weight = row_normalization(self.num_nodes, edge_index, self.weight)
        else:
            raise NotImplementedError
        self.__normed__ = norm

    def in_degrees(self):
        return self.row_ptr[1:] - self.row_ptr[:-1]

    def convert_csr(self):
        if indicator is True:
            self._to_csr()

    def _to_csr(self):
        self.row_ptr, reindex = coo2csr_index(self.row, self.col, num_nodes=self.num_nodes)
        self.col = self.col[reindex]
        self.row = self.row[reindex]
        if self.weight is None:
            self.weight = torch.ones(self.row.shape[0]).to(self.row.device)
        else:
            self.weight = self.weight[reindex]

    def is_symmetric(self):
        return self.__symmetric__

    @property
    def degrees(self):
        if self.row_ptr is not None:
            return self.row_ptr[1:] - self.row_ptr[:-1]
        else:
            edge_index = torch.stack([self.row, self.col])
            return get_degrees(edge_index, num_nodes=self.num_nodes)

    @property
    def edge_index(self):
        if self.row is None:
            self.row, _, _ = csr2coo(self.row_ptr, self.col, self.weight)
        return torch.stack([self.row, self.col])

    @edge_index.setter
    def edge_index(self, edge_index):
        row, col = edge_index
        if self.row is not None and self.row.shape == row.shape:
            return
        self.row, self.col = row, col
        if indicator is True:
            self._to_csr()

    @property
    def row_indptr(self):
        if self.row_ptr is None:
            self._to_csr()
        return self.row_ptr

    @property
    def num_edges(self):
        if self.row is not None:
            return self.row.shape[0]
        elif self.row_ptr is not None:
            return self.row_ptr[-1]
        else:
            return None

    @property
    def num_nodes(self):
        if self.__num_nodes__ is not None:
            return self.__num_nodes__
        elif self.row_ptr is not None:
            return max(torch.max(self.col).item() + 1, self.row_ptr.shape[0] - 1)
        else:
            return torch.max(torch.stack([self.row, self.col])).item() + 1

    @property
    def device(self):
        return self.row.device if self.row is not None else self.row_ptr.device

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __out_repr__(self):
        if self.row is not None:
            info = ["{}={}".format(key, list(self[key].size())) for key in ["edge_index"]]
        else:
            info = ["{}={}".format(key, list(self[key].size())) for key in ["row", "col"] if self[key] is not None]
        info += [
            "edge_{}={}".format(key, list(self[key].size())) for key in ["weight", "attr"] if self[key] is not None
        ]
        return info

    def __getitem__(self, item):
        assert type(item) == str, f"{item} must be str"
        if item[0] == "_" and item[1] != "_":
            # item = re.search("[_]*(.*)", item).group(1)
            item = item[1:]
        return getattr(self, item)

    def __copy__(self):
        result = self.__class__()
        for key in self.keys:
            setattr(result, key, copy.copy(self[key]))
        result.__num_nodes__ = self.__num_nodes__
        return result

    def __deepcopy__(self, memodict={}):
        result = self.__class__()
        memodict[id(self)] = result
        for k in self.keys:
            v = self[k]
            setattr(result, k, copy.deepcopy(v, memodict))
        result.__num_nodes__ = self.__num_nodes__
        return result

    def __repr__(self):
        info = [
            "{}={}".format(key, list(self[key].size()))
            for key in self.keys
            if not key.startswith("__") and self[key] is not None
        ]
        return "{}({})".format(self.__class__.__name__, ", ".join(info))

    def clone(self):
        return Adjacency.from_dict({k: v.clone() for k, v in self})

    @staticmethod
    def from_dict(dictionary):
        r"""Creates a data object from a python dictionary."""
        data = Adjacency()
        for key, item in dictionary.items():
            data[key] = item
        return data


KEY_MAP = {
    "edge_weight": "weight",
    "edge_attr": "attr",
}
EDGE_INDEX = "edge_index"
EDGE_WEIGHT = "edge_weight"
EDGE_ATTR = "edge_attr"
ROW_PTR = "row_indptr"
COL_INDICES = "col_indices"


def is_adj_key_train(key):
    return key.endswith("_train") and is_read_adj_key(key)


def is_adj_key(key):
    return key in ["row", "col", "row_ptr", "attr", "weight"]


def is_read_adj_key(key):
    return sum([x in key for x in [EDGE_INDEX, EDGE_WEIGHT, EDGE_ATTR]]) > 0 or is_adj_key(key)


class Graph(BaseGraph):
    def __init__(self, x=None, y=None, **kwargs):
        super(Graph, self).__init__()
        self.x = x
        self.y = y

        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            elif not is_read_adj_key(key):
                self[key] = item

        num_nodes = x.shape[0] if x is not None else None
        if "edge_index_train" in kwargs:
            self._adj_train = Adjacency(num_nodes=num_nodes)
            for key, item in kwargs.items():
                if is_adj_key_train(key):
                    _key = re.search(r"(.*)_train", key).group(1)
                    if _key.startswith("edge_"):
                        _key = _key.split("edge_")[1]
                    if _key == "index":
                        self._adj_train.edge_index = item
                    else:
                        self._adj_train[_key] = item
        else:
            self._adj_train = None

        self._adj_full = Adjacency(num_nodes=num_nodes)
        for key, item in kwargs.items():
            if is_read_adj_key(key) and not is_adj_key_train(key):
                if key.startswith("edge_"):
                    key = key.split("edge_")[-1]
                if key == "index":
                    self._adj_full.edge_index = item
                else:
                    self._adj_full[key] = item

        self._adj = self._adj_full
        self.__is_train__ = False
        self.__temp_adj_stack__ = list()

    def train(self):
        self.__is_train__ = True
        if self._adj_train is not None:
            self._adj = self._adj_train

    def eval(self):
        self._adj = self._adj_full
        self.__is_train__ = False

    def add_remaining_self_loops(self):
        self._adj_full.add_remaining_self_loops()
        if self._adj_train is not None:
            self._adj_train.add_remaining_self_loops()

    def remove_self_loops(self):
        self._adj_full.remove_self_loops()
        if self._adj_train is not None:
            self._adj_train.remove_self_loops()

    def row_norm(self):
        self._adj.row_norm()

    def sym_norm(self):
        self._adj.sym_norm()

    def is_symmetric(self):
        return self._adj.is_symmetric()

    @contextmanager
    def local_graph(self, key=None):
        self.__temp_adj_stack__.append(self._adj)
        if key is None:
            adj = copy.copy(self._adj)
        else:
            adj = copy.copy(self._adj)
            key = KEY_MAP.get(key, key)
            adj[key] = self._adj[key].clone()
        self._adj = adj
        yield
        del adj
        self._adj = self.__temp_adj_stack__.pop()

    @property
    def edge_index(self):
        return self._adj.edge_index

    @property
    def edge_weight(self):
        if self._adj.weight is None or self._adj.weight.shape[0] != self._adj.col.shape[0]:
            self._adj.weight = torch.ones(self._adj.num_edges, device=self._adj.device)
        return self._adj.weight

    @property
    def edge_attr(self):
        return self._adj.attr

    @edge_index.setter
    def edge_index(self, edge_index):
        row, col = edge_index
        self._adj.row = row
        self._adj.col = col

    @edge_weight.setter
    def edge_weight(self, edge_weight):
        self._adj.weight = edge_weight

    @edge_attr.setter
    def edge_attr(self, edge_attr):
        self._adj.attr = edge_attr

    @property
    def row_indptr(self):
        if self._adj.row_ptr is None:
            self._adj.convert_csr()
        return self._adj.row_ptr

    @property
    def col_indices(self):
        if self._adj.row_ptr is None:
            self._adj.convert_csr()
        return self._adj.col

    @row_indptr.setter
    def row_indptr(self, row_ptr):
        self._adj.row_ptr = row_ptr

    @col_indices.setter
    def col_indices(self, col_indices):
        self._adj.col = col_indices

    @property
    def in_norm(self):
        return self._adj.__in_norm__

    @property
    def out_norm(self):
        return self._adj.__out_norm__

    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def degrees(self):
        return self._adj.degrees

    def __keys__(self):
        keys = [key for key in self.keys if "adj" not in key]
        return keys

    def __old_keys__(self):
        keys = self.__keys__()
        keys += [EDGE_INDEX, EDGE_ATTR]
        return keys

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        if is_adj_key(key):
            return getattr(self._adj, key)
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        if is_adj_key(key):
            self._adj[key] = value
        else:
            setattr(self, key, value)

    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        return self._adj.num_edges

    @property
    def num_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_nodes(self):
        if hasattr(self, "__num_nodes__") and self.__num_nodes__ is not None:
            return self.__num_nodes__
        elif self.x is not None:
            return self.x.shape[0]
        else:
            return self._adj.num_nodes

    @property
    def num_classes(self):
        if self.y is not None:
            return int(torch.max(self.y) + 1) if self.y.dim() == 1 else self.y.shape[-1]

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @staticmethod
    def from_pyg_data(data):
        val = {k: v for k, v in data}
        return Graph(**val)

    def clone(self):
        return Graph.from_dict({k: v.clone() for k, v in self})

    def __repr__(self):
        info = ["{}={}".format(key, list(self[key].size())) for key in self.__keys__() if not key.startswith("_")]
        info += self._adj.__out_repr__()
        return "{}({})".format(self.__class__.__name__, ", ".join(info))

    def sample_adj(self, batch, size=-1, replace=True):
        if sample_adj_c is not None:
            if not torch.is_tensor(batch):
                batch = torch.tensor(batch, dtype=torch.long)
            (row_ptr, col_indices, nodes, edges) = sample_adj_c(self._adj.row_ptr, self._adj.col, batch, size, replace)
        else:
            if not (batch[1:] > batch[:-1]).all():
                batch = batch.sort()[0]
            if torch.is_tensor(batch):
                batch = batch.cpu().numpy()
            if self.__is_train__ and self._adj_train is not None:
                key = "__mx_train__"
            else:
                key = "__mx__"
            if not hasattr(self, key):
                row, col = self._adj.edge_index.numpy()
                val = self.edge_weight.numpy()
                N = self.num_nodes
                self[key] = sp.csr_matrix((val, (row, col)), shape=(N, N))
            adj = self[key][batch, :]

            indptr = adj.indptr
            indices = adj.indices
            if size != -1:
                indptr, indices = self._sample_adj(len(batch), indices, indptr, size)
                indptr = indptr.numpy()
                indices = indices.numpy()
            col_nodes = np.unique(indices)
            _node_idx = np.concatenate([batch, np.setdiff1d(col_nodes, batch)])
            nodes = torch.tensor(_node_idx, dtype=torch.long)

            assoc_dict = {v: i for i, v in enumerate(_node_idx)}

            col_indices = torch.tensor([assoc_dict[i] for i in indices], dtype=torch.long)
            row_ptr = torch.tensor(indptr, dtype=torch.long)

        if row_ptr.shape[0] - 1 < nodes.shape[0]:
            padding = torch.full((nodes.shape[0] - row_ptr.shape[0] + 1,), row_ptr[-1].item(), dtype=row_ptr.dtype)
            row_ptr = torch.cat([row_ptr, padding])
        g = Graph(row_ptr=row_ptr, col=col_indices)
        return nodes, g

    def _sample_adj(self, batch_size, indices, indptr, size):
        if not torch.is_tensor(indices):
            indices = torch.from_numpy(indices)
        if not torch.is_tensor(indptr):
            indptr = torch.from_numpy(indptr)
        assert indptr.shape[0] - 1 == batch_size
        row_counts = (indptr[1:] - indptr[:-1]).long()
        rand = torch.rand(batch_size, size)
        rand = rand * row_counts.view(-1, 1)
        rand = rand.long()

        rand = rand + indptr[:-1].view(-1, 1)
        edge_cols = indices[rand].view(-1)
        row_ptr = torch.arange(0, batch_size * size + size, size)
        return row_ptr, edge_cols

    def csr_subgraph(self, node_idx):
        indptr, indices, nodes, edges = subgraph_c(self._adj.row_ptr, self._adj.col, node_idx.cpu())
        nodes = nodes.to(self._adj.device)
        edge_weight = self.edge_weight[edges]

        data = Graph(row_ptr=indptr, col=indices, weight=edge_weight)
        for key in self.__keys__():
            data[key] = self[key][nodes]
        return data

    def subgraph(self, node_idx):
        if subgraph_c is not None:
            if isinstance(node_idx, list):
                node_idx = torch.as_tensor(node_idx, dtype=torch.long)
            elif isinstance(node_idx, np.ndarray):
                node_idx = torch.from_numpy(node_idx)
            return self.csr_subgraph(node_idx)
        else:
            if isinstance(node_idx, list):
                node_idx = np.array(node_idx)
            elif torch.is_tensor(node_idx):
                node_idx = node_idx.cpu().numpy()
            if self.__is_train__ and self._adj_train is not None:
                key = "__mx_train__"
            else:
                key = "__mx__"
            if not hasattr(self, key):
                row, col = self._adj.edge_index.numpy()
                val = self.edge_weight.numpy()
                N = self.num_nodes
                self[key] = sp.csr_matrix((val, (row, col)), shape=(N, N))
            sub_adj = self[key][node_idx, :][:, node_idx]
            sub_g = Graph()
            sub_g.row_indptr = torch.from_numpy(sub_adj.indptr).long()
            sub_g.col_indices = torch.from_numpy(sub_adj.indices).long()
            sub_g.edge_weight = torch.from_numpy(sub_adj.data)
            for key in self.__keys__():
                sub_g[key] = self[key][node_idx]
            return sub_g.to(self._adj.device)

    def edge_subgraph(self, edge_idx, require_idx=True):
        edge_index = self._adj.edge_index
        edge_index = edge_index[:, edge_idx]
        nodes, new_edge_index = torch.unique(edge_index, return_inverse=True)
        g = Graph(edge_index=new_edge_index)
        for key in self.__keys__():
            g[key] = self[key][nodes]

        if require_idx:
            return g, nodes, edge_idx
        else:
            return g

    @staticmethod
    def from_dict(dictionary):
        r"""Creates a data object from a python dictionary."""
        data = Graph()
        for key, item in dictionary.items():
            data[key] = item
        return data
