from typing import Optional, Tuple
from cogdl import function as BF
from cogdl.backend import BACKEND
if BACKEND == 'jittor':
    from jittor import Module
elif BACKEND == 'torch':
    from torch.nn import Module

from cogdl.utils.graph_utils import symmetric_normalization, row_normalization


class DropFeatures(Module):
    def __init__(self, drop_rate):
        super(DropFeatures, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        return dropout_features(x, self.drop_rate, training=self.training)


class DropEdge(Module):
    def __init__(self, drop_rate: float = 0.5, renorm: Optional[str] = "sym"):
        super(DropEdge, self).__init__()
        self.drop_rate = drop_rate
        self.renorm = renorm

    def forward(self, edge_index, edge_weight=None):
        return dropout_adj(edge_index, edge_weight, self.drop_rate, self.renorm, self.training)


class DropNode(Module):
    def __init__(self, drop_rate=0.5):
        super(DropNode, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        return drop_node(x, self.drop_rate, self.training)


def filter_adj(row, col, edge_attr, mask):
    return (row[mask], col[mask]), None if edge_attr is None else edge_attr[mask]


def dropout_adj(
    edge_index: Tuple,
    edge_weight: Optional[BF.dtype_dict('tensor')] = None,
    drop_rate: float = 0.5,
    renorm: Optional[str] = "sym",
    training: bool = False,
):
    if not training or drop_rate == 0:
        if edge_weight is None:
            edge_weight = BF.ones(edge_index[0].shape[0], device=BF.device(edge_index[0]))
        return edge_index, edge_weight

    if drop_rate < 0.0 or drop_rate > 1.0:
        raise ValueError("Dropout probability has to be between 0 and 1, " "but got {}".format(drop_rate))

    row, col = edge_index
    num_nodes = int(max(row.max(), col.max())) + 1
    self_loop = row == col
    mask = BF.full((row.shape[0],), 1 - drop_rate, dtype=BF.dtype_dict('float'), device=BF.device(row))
    mask = BF.bernoulli(mask).bool()
    mask = self_loop | mask
    edge_index, edge_weight = filter_adj(row, col, edge_weight, mask)
    if renorm == "sym":
        edge_weight = symmetric_normalization(num_nodes, edge_index[0], edge_index[1])
    elif renorm == "row":
        edge_weight = row_normalization(num_nodes, edge_index[0], edge_index[1])
    return edge_index, edge_weight


def dropout_features(x: BF.dtype_dict('tensor'), droprate: float, training=True):
    n = x.shape[1]
    drop_rates = BF.ones(n, device=BF.device(x)) * droprate
    if training:
        masks = BF.bernoulli(1.0 - drop_rates).view(1, -1).expand_as(x)
        masks = BF.to(masks,x)
        masks = BF.to(masks,x)
        x = masks * x
    return x


def drop_node(x, drop_rate=0.5, training=True):
    n = x.shape[0]
    drop_rates = BF.ones(n) * drop_rate
    if training:
        masks = BF.bernoulli(1.0 - drop_rates).unsqueeze(1)
        x = masks.to(x) * x
        x = x / drop_rate
    return x
