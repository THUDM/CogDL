import re

import torch
import numpy as np
import scipy.sparse as sparse


class Data(object):
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item
        self.__adj = None

    @staticmethod
    def from_dict(dictionary):
        r"""Creates a data object from a python dictionary."""
        data = Data()
        for key, item in dictionary.items():
            data[key] = item
        return data

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

    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        for key, item in self("edge_index", "edge_attr"):
            return item.size(self.cat_dim(key, item))
        return None

    @property
    def num_features(self):
        r"""Returns the number of features per node in the graph."""
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_nodes(self):
        if self.x is not None:
            return self.x.shape[0]
        return torch.max(self.edge_index) + 1

    @property
    def num_classes(self):
        if self.y is not None:
            return int(torch.max(self.y) + 1) if self.y.dim() == 1 else self.y.shape[-1]

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    def is_coalesced(self):
        r"""Returns :obj:`True`, if edge indices are ordered and do not contain
        duplicate entries."""
        row, col = self.edge_index
        index = self.num_nodes * row + col
        return row.size(0) == torch.unique(index).size(0)

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, :obj:`func` is applied to all present
        attributes.
        """
        for key, item in self(*keys):
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

    def clone(self):
        return Data.from_dict({k: v.clone() for k, v in self})

    def _build_adj_(self):
        if self.__adj is not None:
            return
        num_edges = self.edge_index.shape[1]
        edge_index_np = self.edge_index.cpu().numpy()
        num_nodes = self.x.shape[0]
        edge_attr_np = np.ones(num_edges)
        self.__adj = sparse.csr_matrix(
            (edge_attr_np, (edge_index_np[0], edge_index_np[1])), shape=(num_nodes, num_nodes)
        )

    def subgraph(self, node_idx):
        """Return the induced node subgraph."""
        if self.__adj is None:
            self._build_adj_()
        if isinstance(node_idx, torch.Tensor):
            node_idx = node_idx.cpu().numpy()
        node_idx = np.unique(node_idx)

        adj = self.__adj[node_idx, :][:, node_idx]
        adj_coo = sparse.coo_matrix(adj)
        row, col = adj_coo.row, adj_coo.col
        edge_attr = torch.from_numpy(adj_coo.data).to(self.x.device)
        edge_index = torch.from_numpy(np.stack([row, col], axis=0)).to(self.x.device).long()
        keys = self.keys
        attrs = {key: self[key][node_idx] for key in keys if "edge" not in key}
        attrs["edge_index"] = edge_index
        if edge_attr is not None:
            attrs["edge_attr"] = edge_attr
        return Data(**attrs)

    def edge_subgraph(self, edge_idx):
        """Return the induced edge subgraph."""
        if isinstance(edge_idx, torch.Tensor):
            edge_idx = edge_idx.cpu().numpy()
        edge_index = self.edge_index.T[edge_idx].cpu().numpy()
        node_idx = np.unique(edge_index)
        idx_dict = {val: key for key, val in enumerate(node_idx)}

        def func(x):
            return [idx_dict[x[0]], idx_dict[x[1]]]

        edge_index = np.array([func(x) for x in edge_index]).transpose()
        edge_index = torch.from_numpy(edge_index).to(self.x.device)
        edge_attr = self.edge_attr[edge_idx] if self.edge_attr else None

        keys = self.keys
        attrs = {key: self[key][node_idx] for key in keys if "edge" not in key}
        attrs["edge_index"] = edge_index
        if edge_attr is not None:
            attrs["edge_attr"] = edge_attr
        return Data(**attrs)

    def sample_adj(self, batch, size=-1, replace=True):
        assert size != 0
        if self.__adj is None:
            self._build_adj_()
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().numpy()

        adj = self.__adj[batch].tocsr()
        batch_size = len(batch)
        if size == -1:
            adj = adj.tocoo()
            row, col = torch.from_numpy(adj.row), torch.from_numpy(adj.col)
            node_idx = torch.unique(col)
        else:
            indices = torch.from_numpy(adj.indices)
            indptr = torch.from_numpy(adj.indptr)
            node_idx, (row, col) = self._sample_adj(batch_size, indices, indptr, size)
        col = col.numpy()
        _node_idx = node_idx.numpy()

        # Reindexing: target nodes are always put at the front
        _node_idx = list(batch) + list(set(_node_idx).difference(set(batch)))
        node_dict = {val: key for key, val in enumerate(_node_idx)}
        new_col = torch.LongTensor([node_dict[i] for i in col])
        edge_index = torch.stack([row.long(), new_col])

        node_idx = torch.Tensor(_node_idx).long().to(self.x.device)
        edge_index = edge_index.long().to(self.x.device)
        return node_idx, edge_index

    def _sample_adj(self, batch_size, indices, indptr, size):
        indptr = indptr
        row_counts = torch.Tensor([indptr[i] - indptr[i - 1] for i in range(1, len(indptr))])

        # if not replace:
        #     edge_cols = [col[indptr[i]: indptr[i+1]] for i in range(len(indptr)-1)]
        #     edge_cols = [np.random.choice(x, min(size, len(x)), replace=False) for x in edge_cols]
        # else:
        rand = torch.rand(batch_size, size)
        rand = rand * row_counts.view(-1, 1)
        rand = rand.long()
        rand = rand + indptr[:-1].view(-1, 1)
        edge_cols = indices[rand].view(-1)
        row = torch.arange(0, batch_size).view(-1, 1).repeat(1, size).view(-1)
        node_idx = torch.unique(edge_cols)

        return node_idx, (row, edge_cols)

    @staticmethod
    def from_pyg_data(data):
        val = {k: v for k, v in data}
        return Data(**val)

    def __repr__(self):
        info = ["{}={}".format(key, list(item.size())) for key, item in self if not key.startswith("_")]
        return "{}({})".format(self.__class__.__name__, ", ".join(info))
