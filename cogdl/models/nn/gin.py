import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add

from .. import BaseModel, register_model
from cogdl.data import DataLoader


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
            self.register_buffer('eps', torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = remove_self_loops(edge_index)
        edge_weight = torch.ones(edge_index.shape[1]) if edge_weight is None else edge_weight
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (x.shape[0], x.shape[0]))
        adj = adj.to(x.device)
        out = (1 + self.eps) * x + torch.spmm(adj, x)
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out


class GINMLP(nn.Module):
    r"""Multilayer perception with batch normalization

    .. math::
        x^{(i+1)} = \sigma(W^{i}x^{(i)})

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    out_feats : int
        Size of each output sample.
    hidden_dim : int
        Size of hidden layer dimension.
    use_bn : bool, optional
        Apply batch normalization if True, default: `True).
    """
    def __init__(self, in_feats, out_feats, hidden_dim, num_layers, use_bn=True, activation=None):
        super(GINMLP, self).__init__()
        self.use_bn = use_bn
        self.nn = nn.ModuleList()
        if use_bn:
            self.bn = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("number of MLP layers should be positive")
        elif num_layers == 1:
            self.nn.append(nn.Linear(in_feats, out_feats))
        else:
            for i in range(num_layers - 1):
                if i == 0:
                    self.nn.append(nn.Linear(in_feats, hidden_dim))
                else:
                    self.nn.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    self.bn.append(nn.BatchNorm1d(hidden_dim))
            self.nn.append(nn.Linear(hidden_dim, out_feats))

    def forward(self, x):
        h = x
        for i in range(self.num_layers - 1):
            h = self.nn[i](h)
            if self.use_bn:
                h = self.bn[i](h)
            h = F.relu(h)
        return self.nn[self.num_layers - 1](h)


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
        parser.add_argument("--epsilon", type=float, default=0.)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--num-mlp-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--train-epsilon", dest="train_epsilon", action="store_false")
        parser.add_argument("--pooling", type=str, default='sum')
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
        random.shuffle(dataset)
        train_size = int(len(dataset) * args.train_ratio)
        test_size = int(len(dataset) * args.test_ratio)
        bs = args.batch_size
        train_loader = DataLoader(dataset[:train_size], batch_size=bs)
        test_loader = DataLoader(dataset[-test_size:], batch_size=bs)
        if args.train_ratio + args.test_ratio < 1:
            valid_loader = DataLoader(dataset[train_size:-test_size], batch_size=bs)
        else:
            valid_loader = test_loader
        return train_loader, valid_loader, test_loader

    def __init__(self,
                 num_layers,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_mlp_layers,
                 eps=0,
                 pooling='sum',
                 train_eps=False,
                 dropout=0.5,
                 ):
        super(GIN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            if i == 0:
                mlp = GINMLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers)
            else:
                mlp = GINMLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers)
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
        layer_rep = [h]
        for i in range(self.num_layers-1):
            h = self.gin_layers[i](h, batch.edge_index)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            layer_rep.append(h)

        final_score = 0
        for i in range(self.num_layers):
            # pooled = self.pooling(layer_rep[i], batch, dim=0)
            pooled = scatter_add(layer_rep[i], batch.batch, dim=0)
            final_score += self.dropout(self.linear_prediction[i](pooled))
        final_score = F.softmax(final_score, dim=-1)
        if batch.y is not None:
            loss = self.loss(final_score, batch.y)
            return final_score, loss
        return final_score, None

    def loss(self, output, label=None):
        return self.criterion(output, label)
