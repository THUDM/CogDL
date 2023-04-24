import re
import copy
from contextlib import contextmanager
import scipy.sparse as sp
import networkx as nx

import torch
import numpy as np
from cogdl.utils import (
    csr2coo,
    coo2csr_index,
    add_remaining_self_loops,
    symmetric_normalization,
    row_normalization,
    get_degrees,
)
from cogdl.utils import RandomWalker
from cogdl.operators.sample import sample_adj_c, subgraph_c

subgraph_c = None  # noqa: F811


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
        # return len(self.keys)
        return 1

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
        return self.__num_nodes__ if bool(re.search("(index|face)", key)) else 0

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
    def __init__(self, row=None, col=None, row_ptr=None, weight=None, attr=None, num_nodes=None, types=None, **kwargs):
        super(Adjacency, self).__init__()
        self.row = row
        self.col = col
        self.row_ptr = row_ptr
        self.weight = weight
        self.attr = attr
        self.types = types
        self.__num_nodes__ = num_nodes
        self.__normed__ = None
        self.__in_norm__ = self.__out_norm__ = None
        self.__symmetric__ = True
        for key, item in kwargs.items():
            self[key] = item

    def set_weight(self, weight):
        self.weight = weight
        self.__normed__ = None
        self.__in_norm__ = self.__out_norm__ = None
        self.__symmetric__ = False

    def get_weight(self, indicator=None):
        """If `indicator` is not None, the normalization will not be implemented"""
        if self.weight is None or self.weight.shape[0] != self.col.shape[0]:
            self.weight = torch.ones(self.num_edges, device=self.device)
        weight = self.weight
        if indicator is not None:
            return weight

        if self.__in_norm__ is not None:
            if self.row is None:
                num_nodes = self.row_ptr.size(0) - 1
                row = torch.arange(num_nodes, device=self.device)
                row_count = self.row_ptr[1:] - self.row_ptr[:-1]
                self.row = row.repeat_interleave(row_count)
            weight = self.__in_norm__[self.row].view(-1)
        if self.__out_norm__ is not None:
            weight = self.__out_norm__[self.col].view(-1)
        return weight

    def add_remaining_self_loops(self):
        if self.attr is not None and len(self.attr.shape) == 1:
            edge_index, weight_attr = add_remaining_self_loops(
                (self.row, self.col), edge_weight=self.attr, fill_value=0, num_nodes=self.num_nodes
            )
            self.row, self.col = edge_index
            self.attr = weight_attr
            self.weight = torch.ones_like(self.row).float()
        else:
            edge_index, self.weight = add_remaining_self_loops(
                (self.row, self.col), fill_value=1, num_nodes=self.num_nodes
            )
            self.row, self.col = edge_index
            self.attr = None
        self.row_ptr, reindex = coo2csr_index(self.row, self.col, num_nodes=self.num_nodes)
        self.row = self.row[reindex]
        self.col = self.col[reindex]

    def padding_self_loops(self):
        device = self.row.device
        row, col = torch.arange(self.num_nodes, device=device), torch.arange(self.num_nodes, device=device)
        self.row = torch.cat((self.row, row))
        self.col = torch.cat((self.col, col))

        if self.weight is not None:
            values = torch.zeros(self.num_nodes, device=device) + 0.01
            self.weight = torch.cat((self.weight, values))
        if self.attr is not None:
            attr = torch.zeros(self.num_nodes, device=device)
            self.attr = torch.cat((self.attr, attr))
        self.row_ptr, reindex = coo2csr_index(self.row, self.col, num_nodes=self.num_nodes)
        self.row = self.row[reindex]
        self.col = self.col[reindex]

    def remove_self_loops(self):
        mask = self.row == self.col
        inv_mask = ~mask
        self.row = self.row[inv_mask]
        self.col = self.col[inv_mask]
        for item in self.__attr_keys__():
            if self[item] is not None:
                self[item] = self[item][inv_mask]

        self.convert_csr()

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

    def col_norm(self):
        if self.row is None:
            self.generate_normalization("col")
        else:
            self.normalize_adj("col")
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
        elif norm == "col":
            self.row, _, _ = csr2coo(self.row_ptr, self.col, self.weight)
            self.weight = row_normalization(self.num_nodes, self.col, self.row, self.weight)
        else:
            raise NotImplementedError
        self.__normed__ = norm

    def normalize_adj(self, norm="sym"):
        if self.__normed__:
            return
        if self.weight is None or self.weight.shape[0] != self.col.shape[0]:
            self.weight = torch.ones(self.num_edges, device=self.device)

        if norm == "sym":
            self.weight = symmetric_normalization(self.num_nodes, self.row, self.col, self.weight)
        elif norm == "row":
            self.weight = row_normalization(self.num_nodes, self.row, self.col, self.weight)
        elif norm == "col":
            self.weight = row_normalization(self.num_nodes, self.col, self.row, self.weight)
        else:
            raise NotImplementedError
        self.__normed__ = norm

    def convert_csr(self):
        self._to_csr()

    def _to_csr(self):
        self.row_ptr, reindex = coo2csr_index(self.row, self.col, num_nodes=self.num_nodes)
        self.col = self.col[reindex]
        self.row = self.row[reindex]
        for key in self.__attr_keys__():
            if key == "weight" and self[key] is None:
                self.weight = torch.ones(self.row.shape[0]).to(self.row.device)
            if self[key] is not None:
                self[key] = self[key][reindex]

    def is_symmetric(self):
        return self.__symmetric__

    def set_symmetric(self, val):
        assert val in [True, False]
        self.__symmetric__ = val

    def degrees(self, node_idx=None):
        if self.row_ptr is not None:
            degs = (self.row_ptr[1:] - self.row_ptr[:-1]).float()
            if node_idx is not None:
                return degs[node_idx]
            return degs
        else:
            return get_degrees(self.row, self.col, num_nodes=self.num_nodes)

    @property
    def edge_index(self):
        if self.row is None:
            self.row, _, _ = csr2coo(self.row_ptr, self.col, self.weight)
        return self.row, self.col

    @edge_index.setter
    def edge_index(self, edge_index):
        row, col = edge_index
        # if self.row is not None and self.row.shape == row.shape:
        #     return
        self.row, self.col = row, col
        # self.convert_csr()
        self.row_ptr = None

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
        if self.row_ptr is not None:
            return self.row_ptr.shape[0] - 1
        else:
            self.__num_nodes__ = max(self.row.max().item(), self.col.max().item()) + 1
            return self.__num_nodes__

    @property
    def row_ptr_v(self):
        return self.row_ptr

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
            info = ["{}={}".format("edge_index", [2] + list(self.row.size()))]
        else:
            info = ["{}={}".format(key, list(self[key].size())) for key in ["row", "col"] if self[key] is not None]
        attr_key = self.__attr_keys__()
        info += ["edge_{}={}".format(key, list(self[key].size())) for key in attr_key if self[key] is not None]
        return info

    def __getitem__(self, item):
        assert type(item) == str, f"{item} must be str"
        if item[0] == "_" and item[1] != "_":
            # item = re.search("[_]*(.*)", item).group(1)
            item = item[1:]
        if item.startswith("edge_") and item != "edge_index":
            item = item[5:]
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise KeyError(f"{item} not in Adjacency")

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

    def __attr_keys__(self):
        return [x for x in self.keys if "row" not in x and "col" not in x]

    def clone(self):
        return Adjacency.from_dict({k: v.clone() for k, v in self})

    def to_scipy_csr(self):
        data = self.get_weight().cpu().numpy()
        num_nodes = int(self.num_nodes)
        if self.row_ptr is None:
            row = self.row.cpu().numpy()
            col = self.col.cpu().numpy()
            mx = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        else:
            row_ptr = self.row_ptr.cpu().numpy()
            col_ind = self.col.cpu().numpy()
            mx = sp.csr_matrix((data, col_ind, row_ptr), shape=(num_nodes, num_nodes))
        return mx

    def to_networkx(self, weighted=True):
        gnx = nx.Graph()
        gnx.add_nodes_from(np.arange(self.num_nodes))
        row, col = self.edge_index
        row = row.tolist()
        col = col.tolist()

        if weighted:
            weight = self.get_weight().tolist()
            gnx.add_weighted_edges_from([(row[i], col[i], weight[i]) for i in range(len(row))])
        else:
            edges = torch.stack((row, col)).cpu().numpy().transpose()
            gnx.add_edges_from(edges)
        return gnx

    def random_walk(self, seeds, length=1, restart_p=0.0, parallel=True):
        if not hasattr(self, "__walker__"):
            scipy_adj = self.to_scipy_csr()
            self.__walker__ = RandomWalker(scipy_adj)
        return self.__walker__.walk(seeds, length, restart_p=restart_p, parallel=parallel)

    @staticmethod
    def from_dict(dictionary):
        r"""Creates a data object from a python dictionary."""
        data = Adjacency()
        for key, item in dictionary.items():
            data[key] = item
        return data


KEY_MAP = {"edge_weight": "weight", "edge_attr": "attr", "edge_types": "types"}
EDGE_INDEX = "edge_index"
EDGE_WEIGHT = "edge_weight"
EDGE_ATTR = "edge_attr"
ROW_PTR = "row_indptr"
COL_INDICES = "col_indices"


def is_adj_key_train(key):
    return key.endswith("_train") and is_read_adj_key(key)


def is_adj_key(key):
    return key in ["row", "col", "row_ptr", "attr", "weight", "types"] or key.startswith("edge_")


def is_read_adj_key(key):
    return sum([x in key for x in [EDGE_INDEX, EDGE_WEIGHT, EDGE_ATTR]]) > 0 or is_adj_key(key)


class Graph(BaseGraph):
    def __init__(self, x=None, y=None, **kwargs):
        super(Graph, self).__init__()
        if x is not None:
            if not torch.is_tensor(x):
                raise ValueError("Node features must be Tensor")
        self.x = x
        self.y = y
        self.grb_adj = None
        num_nodes = x.shape[0] if x is not None else None

        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
                num_nodes = item
            elif key == "grb_adj":
                self.grb_adj = item
            elif not is_read_adj_key(key):
                self[key] = item

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
        self.__temp_storage__ = dict()

    def train(self):
        self.__is_train__ = True
        if self._adj_train is not None:
            self._adj = self._adj_train
        return self

    def eval(self):
        self._adj = self._adj_full
        self.__is_train__ = False
        return self

    def add_remaining_self_loops(self):
        self._adj_full.add_remaining_self_loops()
        if self._adj_train is not None:
            self._adj_train.add_remaining_self_loops()
        return self

    def padding_self_loops(self):
        self._adj.padding_self_loops()
        return self

    def remove_self_loops(self):
        self._adj_full.remove_self_loops()
        if self._adj_train is not None:
            self._adj_train.remove_self_loops()
        return self

    def row_norm(self):
        self._adj.row_norm()

    def col_norm(self):
        self._adj.col_norm()

    def sym_norm(self):
        self._adj.sym_norm()

    def normalize(self, key="sym"):
        assert key in ["row", "sym", "col"], "Support row/col/sym normalization"
        getattr(self, f"{key}_norm")()

    def is_symmetric(self):
        return self._adj.is_symmetric()

    def set_symmetric(self):
        self._adj.set_symmetric(True)

    def set_asymmetric(self):
        self._adj.set_symmetric(False)

    def is_inductive(self):
        return self._adj_train is not None

    def mask2nid(self, split):
        mask = getattr(self, f"{split}_mask")
        if mask is not None:
            if mask.dtype is torch.bool:
                return torch.where(mask)[0]
            return mask

    @property
    def train_nid(self):
        return self.mask2nid("train")

    @property
    def val_nid(self):
        return self.mask2nid("val")

    @property
    def test_nid(self):
        return self.mask2nid("test")

    @contextmanager
    def local_graph(self):
        self.__temp_adj_stack__.append(self._adj)
        adj = copy.copy(self._adj)
        others = [(key, val) for key, val in self.__dict__.items() if not key.startswith("__") and "adj" not in key]
        self._adj = adj
        yield
        del adj
        self._adj = self.__temp_adj_stack__.pop()
        for key, val in others:
            self[key] = val

    @property
    def edge_index(self):
        return self._adj.edge_index

    @property
    def edge_weight(self):
        """Return actual edge_weight"""
        return self._adj.get_weight()

    @property
    def raw_edge_weight(self):
        """Return edge_weight without __in_norm__ and __out_norm__, only used for SpMM"""
        return self._adj.get_weight("raw")

    @property
    def edge_attr(self):
        return self._adj.attr

    @property
    def edge_types(self):
        return self._adj.types

    @edge_index.setter
    def edge_index(self, edge_index):
        if edge_index is None:
            self._adj.row = None
            self._adj.col = None
            self.__num_nodes__ = 0
        else:
            row, col = edge_index
            if self._adj.row is not None and row.shape[0] != self._adj.row.shape[0]:
                self._adj.row_ptr = None
            self._adj.row = row
            self._adj.col = col
            if self.x is not None:
                self._adj.__num_nodes__ = self.x.shape[0]
                self.__num_nodes__ = self.x.shape[0]
            else:
                self.__num_nodes__ = None

    @edge_weight.setter
    def edge_weight(self, edge_weight):
        self._adj.set_weight(edge_weight)

    @edge_attr.setter
    def edge_attr(self, edge_attr):
        self._adj.attr = edge_attr

    @edge_types.setter
    def edge_types(self, edge_types):
        self._adj.types = edge_types

    @property
    def row_indptr(self):
        return self._adj.row_indptr

    @property
    def col_indices(self):
        if self._adj.row_ptr is None:
            self._adj._to_csr()
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

    @property
    def device(self):
        return self._adj.device

    def degrees(self):
        return self._adj.degrees()

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
            if key[0] == "_" and key[1] != "_":
                key = key[1:]
            if key.startswith("edge_") and key != "edge_index":
                key = key[5:]
            return getattr(self._adj, key)
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        if is_adj_key(key):
            if key[0] == "_" and key[1] != "_":
                key = key[1:]
            if key.startswith("edge_") and key != "edge_index":
                key = key[5:]
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

    def store(self, key):
        if hasattr(self, key) and not callable(getattr(self, key)):
            self.__temp_storage__[key] = copy.deepcopy(getattr(self, key))
        if hasattr(self._adj, key) and not callable(getattr(self._adj, key)):
            self.__temp_storage__[key] = copy.deepcopy(getattr(self._adj, key))

    def restore(self, key):
        if key in self.__temp_storage__:
            if hasattr(self, key) and not callable(getattr(self, key)):
                setattr(self, key, self.__temp_storage__[key])
            elif hasattr(self._adj, key) and not callable(getattr(self._adj, key)):
                self(self._adj, key, self.__temp_storage__[key])
            self.__temp_storage__.pop(key)

    def __delitem__(self, key):
        if hasattr(self, key):
            self[key] = None

    def __repr__(self):
        info = [
            "{}={}".format(key, list(self[key].size()))
            for key in self.__keys__()
            if not key.startswith("_") and hasattr(self[key], "size")
        ]
        info += self._adj.__out_repr__()
        return "{}({})".format(self.__class__.__name__, ", ".join(info))

    def sample_adj(self, batch, size=-1, replace=True):
        if sample_adj_c is not None:
            if not torch.is_tensor(batch):
                batch = torch.tensor(batch, dtype=torch.long)
            (row_ptr, col_indices, nodes, edges) = sample_adj_c(
                self.row_indptr, self.col_indices, batch, size, replace
            )
        else:
            if torch.is_tensor(batch):
                batch = batch.cpu().numpy()
            if self.__is_train__ and self._adj_train is not None:
                key = "__mx_train__"
            else:
                key = "__mx__"
            if not hasattr(self, key):
                row, col = self._adj.row.numpy(), self._adj.col.numpy()
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

    def csr_subgraph(self, node_idx, keep_order=False):
        if self._adj.row_ptr_v is None:
            self._adj._to_csr()
        if torch.is_tensor(node_idx):
            node_idx = node_idx.cpu()
        else:
            node_idx = torch.as_tensor(node_idx)

        if not keep_order:
            node_idx = torch.unique(node_idx)
        indptr, indices, nodes, edges = subgraph_c(self._adj.row_ptr, self._adj.col, node_idx)
        nodes_idx = node_idx.to(self._adj.device)

        data = Graph(row_ptr=indptr, col=indices)
        for key in self.__keys__():
            data[key] = self[key][nodes_idx]
        for key in self._adj.keys:
            if "row" in key or "col" in key:
                continue
            if key.startswith("__"):
                continue
            data._adj[key] = self._adj[key][edges]
        data.num_nodes = node_idx.shape[0]
        data.edge_weight = None
        return data

    def subgraph(self, node_idx, keep_order=False):
        if subgraph_c is not None:
            if isinstance(node_idx, list):
                node_idx = torch.as_tensor(node_idx, dtype=torch.long)
            elif isinstance(node_idx, np.ndarray):
                node_idx = torch.from_numpy(node_idx)
            return self.csr_subgraph(node_idx, keep_order)
        else:
            if isinstance(node_idx, list):
                node_idx = np.array(node_idx, dtype=np.int64)
            elif torch.is_tensor(node_idx):
                node_idx = node_idx.long().cpu().numpy()
            if self.__is_train__ and self._adj_train is not None:
                key = "__mx_train__"
            else:
                key = "__mx__"
            if not hasattr(self, key):
                row = self._adj.row.numpy()
                col = self._adj.col.numpy()
                val = self.edge_weight.numpy()
                N = self.num_nodes
                self[key] = sp.csr_matrix((val, (row, col)), shape=(N, N))
            sub_adj = self[key][node_idx, :][:, node_idx].tocoo()
            sub_g = Graph()
            # sub_g.row_indptr = torch.from_numpy(sub_adj.indptr).long()
            # sub_g.col_indices = torch.from_numpy(sub_adj.indices).long()
            row = torch.from_numpy(sub_adj.row).long()
            col = torch.from_numpy(sub_adj.col).long()
            sub_g.edge_index = (row, col)
            sub_g.edge_weight = torch.from_numpy(sub_adj.data)
            sub_g.num_nodes = len(node_idx)
            for key in self.__keys__():
                sub_g[key] = self[key][node_idx]
            sub_g._adj._to_csr()
            return sub_g.to(self._adj.device)

    def edge_subgraph(self, edge_idx, require_idx=True):
        row, col = self._adj.edge_index
        row = row[edge_idx]
        col = col[edge_idx]
        edge_index = torch.stack([row, col])
        nodes, new_edge_index = torch.unique(edge_index, return_inverse=True)
        g = Graph(edge_index=new_edge_index)
        for key in self.__keys__():
            g[key] = self[key][nodes]

        if require_idx:
            return g, nodes, edge_idx
        else:
            return g

    def random_walk(self, seeds, max_nodes_per_seed, restart_p=0.0, parallel=True):
        return self._adj.random_walk(seeds, max_nodes_per_seed, restart_p, parallel)

    def random_walk_with_restart(self, seeds, max_nodes_per_seed, restart_p=0.0, parallel=True):
        return self._adj.random_walk(seeds, max_nodes_per_seed, restart_p, parallel)

    def to_scipy_csr(self):
        return self._adj.to_scipy_csr()

    def to_networkx(self):
        return self._adj.to_networkx()

    @staticmethod
    def from_dict(dictionary):
        r"""Creates a data object from a python dictionary."""
        data = Graph()
        for key, item in dictionary.items():
            data[key] = item
        return data

    def nodes(self):
        return torch.arange(self.num_nodes)

    def set_grb_adj(self, adj):
        self.grb_adj = adj

    # @property
    # def requires_grad(self):
    #     return False
    #
    # @requires_grad.setter
    # def requires_grad(self, x):
    #     print(f"Set `requires_grad` to {x}")
