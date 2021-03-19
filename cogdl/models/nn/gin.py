import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .mlp import MLP
from cogdl.data import DataLoader
from cogdl.utils import spmm


def split_dataset_general(dataset, args):
    droplast = args.model == "diffpool"

    train_size = int(len(dataset) * args.train_ratio)
    test_size = int(len(dataset) * args.test_ratio)
    index = list(range(len(dataset)))
    random.shuffle(index)

    train_index = index[:train_size]
    test_index = index[-test_size:]

    bs = args.batch_size
    train_loader = DataLoader([dataset[i] for i in train_index], batch_size=bs, drop_last=droplast)
    test_loader = DataLoader([dataset[i] for i in test_index], batch_size=bs, drop_last=droplast)
    if args.train_ratio + args.test_ratio < 1:
        val_index = index[train_size:-test_size]
        valid_loader = DataLoader([dataset[i] for i in val_index], batch_size=bs, drop_last=droplast)
    else:
        valid_loader = test_loader
    return train_loader, valid_loader, test_loader


class GINLayer(nn.Module):
    r"""Graph Isomorphism Network layer from paper `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{sum}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable layer function)
        layer or function applied to update node feature
    eps : float32, optional
        Initial `\epsilon` value.
    train_eps : bool, optional
        If True, `\epsilon` will be a learnable parameter.
    """

    def __init__(self, apply_func=None, eps=0, train_eps=True):
        super(GINLayer, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, graph, x):
        # edge_index, _ = remove_self_loops()
        # edge_weight = torch.ones(edge_index.shape[1]).to(x.device) if edge_weight is None else edge_weight
        # adj = torch.sparse_coo_tensor(edge_index, edge_weight, (x.shape[0], x.shape[0]))
        # adj = adj.to(x.device)
        # out = (1 + self.eps) * x + torch.spmm(adj, x)
        out = (1 + self.eps) * x + spmm(graph, x)
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out


@register_model("gin")
class GIN(BaseModel):
    r"""Graph Isomorphism Network from paper `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`__.

    Args:
        num_layers : int
            Number of GIN layers
        in_feats : int
            Size of each input sample
        out_feats : int
            Size of each output sample
        hidden_dim : int
            Size of each hidden layer dimension
        num_mlp_layers : int
            Number of MLP layers
        eps : float32, optional
            Initial `\epsilon` value, default: ``0``
        pooling : str, optional
            Aggregator type to use, default:　``sum``
        train_eps : bool, optional
            If True, `\epsilon` will be a learnable parameter, default: ``True``
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--epsilon", type=float, default=0.0)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--num-mlp-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--train-epsilon", dest="train_epsilon", action="store_false")
        parser.add_argument("--pooling", type=str, default="sum")
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_layers,
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_mlp_layers,
            args.epsilon,
            args.pooling,
            args.train_epsilon,
            args.dropout,
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        return split_dataset_general(dataset, args)

    def __init__(
        self,
        num_layers,
        in_feats,
        out_feats,
        hidden_dim,
        num_mlp_layers,
        eps=0,
        pooling="sum",
        train_eps=False,
        dropout=0.5,
    ):
        super(GIN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            if i == 0:
                mlp = MLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            self.gin_layers.append(GINLayer(mlp, eps, train_eps))
            self.batch_norm.append(nn.BatchNorm1d(hidden_dim))

        self.linear_prediction = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.linear_prediction.append(nn.Linear(in_feats, out_feats))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_feats))
        self.dropout = nn.Dropout(dropout)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        h = batch.x
        device = h.device
        batchsize = int(torch.max(batch.batch)) + 1

        layer_rep = [h]
        for i in range(self.num_layers - 1):
            h = self.gin_layers[i](batch, h)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            layer_rep.append(h)

        final_score = 0

        for i in range(self.num_layers):
            # pooled = self.pooling(layer_rep[i], batch, dim=0)
            # pooled = scatter_add(layer_rep[i], batch.batch, dim=0)
            hsize = layer_rep[i].shape[1]
            output = torch.zeros(batchsize, layer_rep[i].shape[1]).to(device)
            pooled = output.scatter_add_(dim=0, index=batch.batch.view(-1, 1).repeat(1, hsize), src=layer_rep[i])
            final_score += self.dropout(self.linear_prediction[i](pooled))
        return final_score
