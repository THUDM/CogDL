from typing import Optional, Type, Any
import scipy.sparse as sp

import torch
import torch.nn as nn

from cogdl.trainers.base_trainer import BaseTrainer
from cogdl.utils import spmm_adj

try:
    from cogdl.layers.spmm_layer import csrspmm
except Exception as e:
    print(e)
    csrspmm = None


class BaseLayer(nn.Module):
    _cache = dict()

    def __init__(self):
        super(BaseLayer, self).__init__()

    def get_csr_ind(self, edge_index, edge_attr=None, num_nodes=None):
        flag = str(edge_index.shape[1])
        if flag not in BaseLayer._cache:
            if num_nodes is None:
                num_nodes = torch.max(edge_index) + 1
            if edge_attr is None:
                edge_attr = torch.ones(edge_index.shape[1], device=edge_index.device)
            BaseLayer._cache[flag] = get_csr_from_edge_index(edge_index, edge_attr, size=(num_nodes, num_nodes))
        cache = BaseLayer._cache[flag]
        colptr = cache["colptr"]
        row_indices = cache["row_indices"]
        csr_data = cache["csr_data"]
        rowptr = cache["rowptr"]
        col_indices = cache["col_indices"]
        csc_data = cache["csc_data"]
        return colptr, row_indices, csr_data, rowptr, col_indices, csc_data

    def spmm(self, edge_index, edge_attr, x):
        if csrspmm is not None and str(x.device) != "cpu":
            (
                colptr, row_indices, csr_data,
                rowptr, col_indices, csc_data
            ) = self.get_csr_ind(edge_index, edge_attr, x.shape[0])
            x = csrspmm(rowptr, col_indices, colptr, row_indices, x, csr_data, csc_data)
        else:
            x = spmm_adj(edge_index, edge_attr, x, x.shape[0])
        return x


class BaseModel(BaseLayer):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new model instance."""
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = ""
        self.loss_fn = None
        self.evaluator = None

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, data):
        return self.forward(data.x, data.edge_index)

    def node_classification_loss(self, data, mask=None):
        if mask is None:
            mask = data.train_mask
        assert mask.shape[0] == data.y.shape[0]
        edge_index = data.edge_index_train if hasattr(data, "edge_index_train") and self.training else data.edge_index
        pred = self.forward(data.x, edge_index)
        return self.loss_fn(pred[mask], data.y[mask])

    def graph_classification_loss(self, batch):
        pred = self.forward(batch)
        return self.loss_fn(pred, batch.y)

    @staticmethod
    def get_trainer(task: Any, args: Any) -> Optional[Type[BaseTrainer]]:
        return None

    def set_device(self, device):
        self.device = device

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn


def get_csr_from_edge_index(edge_index, edge_attr, size):
    device = edge_index.device
    _edge_index = edge_index.cpu().numpy()
    _edge_attr = edge_attr.cpu().numpy()
    num_nodes = size[0]

    adj = sp.csr_matrix((_edge_attr, (_edge_index[0], _edge_index[1])), shape=(num_nodes, num_nodes))
    colptr = torch.as_tensor(adj.indptr, dtype=torch.int32).to(device)
    row_indices = torch.as_tensor(adj.indices, dtype=torch.int32).to(device)
    csr_data = torch.as_tensor(adj.data, dtype=torch.float).to(device)
    adj = adj.tocsc()
    rowptr = torch.as_tensor(adj.indptr, dtype=torch.int32).to(device)
    col_indices = torch.as_tensor(adj.indices, dtype=torch.int32).to(device)
    csc_data = torch.as_tensor(adj.data, dtype=torch.float).to(device)
    cache = {
        "colptr": colptr,
        "row_indices": row_indices,
        "csr_data": csr_data,
        "rowptr": rowptr,
        "col_indices": col_indices,
        "csc_data": csc_data
    }
    return cache


def coo2csr(edge_index, edge_attr, num_nodes=None):
    if num_nodes is None:
        num_nodes = torch.max(edge_index) + 1
    device = edge_index[0].device
    sorted_index = torch.argsort(edge_index[0])
    sorted_index = sorted_index.long()
    edge_index = edge_index[:, sorted_index]
    edge_attr = edge_attr[sorted_index]
    indices = edge_index[1]

    row = edge_index[0]
    indptr = torch.zeros(num_nodes+1, dtype=torch.int32, device=device)
    elements, counts = torch.unique(row, return_counts=True)
    elements = elements.long() + 1
    indptr[elements] = counts.to(indptr.dtype)
    indptr = indptr.cumsum(dim=0)
    return indptr.int(), indices.int(), edge_attr


def coo2csc(edge_index, edge_attr):
    edge_index = torch.stack([edge_index[1], edge_index[0]])
    return coo2csr(edge_index, edge_attr)
