import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn import GINConv

from .. import BaseModel, register_model
from cogdl.data import DataLoader

# def broadcast(src, other, dim):
#     if dim < 0:
#         dim = other.dim() + dim
#     if src.dim() == 1:
#         for _ in range(0, dim):
#             src = src.unsqueeze(0)
#     for _ in range(src.dim(), other.dim()):
#         src = src.unsqueeze(-1)
#     src = src.expand_as(other)
#     return src
#
#
# def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
#     index = broadcast(index, src, dim)
#     if out is None:
#         size = list(src.shape)
#         if dim_size is not None:
#             size[dim] = dim_size
#         elif index.numel() == 0:
#             size[dim] = 0
#         else:
#             size[dim] = int(index.max()) + 1
#         out = torch.zeros(size, dtype=src.dtype, device=src.device)
#         return out.scatter_add_(dim, index, src)
#     else:
#         return out.scatter_add_(dim, index, src)



class GINLayer(nn.Module):
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
        adj = adj.cuda()
        out = (1 + self.eps) * x + torch.spmm(adj, x)
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out


class GINMLP(nn.Module):
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
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--epsilon", type=float, default=0.)
        parser.add_argument("--hidden-size", type=int, default=32)
        parser.add_argument("--num-layers", type=int, default=5)
        parser.add_argument("--num-mlp-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--train-epsilon", type=bool, default=True)
        parser.add_argument("--pooling", type=str, default='sum')
        parser.add_argument("--batch-size", type=int, default=128)

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
        test_index = random.sample(range(len(dataset)), len(dataset)//10)
        train_index = [x for x in range(len(dataset)) if x not in test_index]

        train_dataset = [dataset[i] for i in train_index]
        test_dataset = [dataset[i] for i in test_index]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        return train_loader, test_loader, test_loader

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

        # if pooling == 'sum':
        #     pooling_layer = scatter_add
        # elif pooling == 'mean':
        #     raise NotImplementedError
        # elif pooling == 'max':
        #     raise NotImplementedError
        # else:
        #     raise ValueError("pooling type should be 'sum', 'mean' or 'max'")
        # self.pooling = pooling_layer

        self.linear_prediction = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.linear_prediction.append(nn.Linear(in_feats, out_feats))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_feats))
        self.dropout = nn.Dropout(dropout)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch, edge_weight=None, label=None):
        h = x
        layer_rep = [h]
        for i in range(self.num_layers-1):
            h = self.gin_layers[i](h, edge_index)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            layer_rep.append(h)

        final_score = 0
        for i in range(self.num_layers):
            # pooled = self.pooling(layer_rep[i], batch, dim=0)
            pooled = scatter_add(layer_rep[i], batch, dim=0)
            final_score += self.dropout(self.linear_prediction[i](pooled))
        final_score = F.softmax(final_score, dim=-1)
        if label is not None:
            loss = self.loss(final_score, label)
            return final_score, loss
        return final_score, None

    def loss(self, output, label=None):
        return self.criterion(output, label)
