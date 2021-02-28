import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import register_model, BaseModel
from cogdl.utils import symmetric_normalization, add_remaining_self_loops, spmm


class GCNIILayer(nn.Module):
    def __init__(self, n_channels, alpha=0.1, beta=1, residual=False):
        super(GCNIILayer, self).__init__()
        self.n_channels = n_channels
        self.alpha = alpha
        self.beta = beta
        self.residual = residual
        self.weight = nn.Parameter(torch.FloatTensor(n_channels, n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.n_channels)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_attr, init_x):
        """Symmetric normalization"""
        hidden = spmm(edge_index, edge_attr, x)
        hidden = (1 - self.alpha) * hidden + self.alpha * init_x
        h = self.beta * torch.matmul(hidden, self.weight) + (1 - self.beta) * hidden
        if self.residual:
            h = h + x
        return h


@register_model("gcnii")
class GCNII(BaseModel):
    """
    Implementation of GCNII in paper `"Simple and Deep Graph Convolutional Networks"` <https://arxiv.org/abs/2007.02133>.

    Parameters
    -----------
    in_feats : int
        Size of each input sample
    hidden_size : int
        Size of each hidden unit
    out_feats : int
        Size of each out sample
    num_layers : int
    dropout : float
    alpha : float
        Parameter of initial residual connection
    lmbda : float
        Parameter of identity mapping
    wd1 : float
        Weight-decay for Fully-connected layers
    wd2 : float
        Weight-decay for convolutional layers
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-layers", type=int, default=64)
        parser.add_argument("--lambda", dest="lmbda", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--wd1", type=float, default=0.01)
        parser.add_argument("--wd2", type=float, default=5e-4)
        parser.add_argument("--residual", action="store_true")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feats=args.num_features,
            hidden_size=args.hidden_size,
            out_feats=args.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=args.alpha,
            lmbda=args.lmbda,
            wd1=args.wd1,
            wd2=args.wd2,
            residual=args.residual,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout=0.5,
        alpha=0.1,
        lmbda=1,
        wd1=0.0,
        wd2=0.0,
        residual=False,
    ):
        super(GCNII, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_features=in_feats, out_features=hidden_size))
        self.fc_layers.append(nn.Linear(in_features=hidden_size, out_features=out_feats))

        self.dropout = dropout
        self.alpha = alpha
        self.lmbda = lmbda
        self.wd1 = wd1
        self.wd2 = wd2

        self.layers = nn.ModuleList(
            GCNIILayer(hidden_size, self.alpha, math.log(self.lmbda / (i + 1) + 1), residual) for i in range(num_layers)
        )
        self.activation = F.relu

        self.fc_parameters = list(self.fc_layers.parameters())
        self.conv_parameters = list(self.layers.parameters())
        self.cache = dict()

    def forward(self, x, edge_index, edge_attr=None):
        _attr = str(edge_index.shape[1])
        if _attr not in self.cache:
            edge_index, edge_attr = add_remaining_self_loops(
                edge_index=edge_index,
                edge_weight=torch.ones(edge_index.shape[1]).to(x.device),
                fill_value=1,
                num_nodes=x.shape[0],
            )
            edge_attr = symmetric_normalization(x.shape[0], edge_index, edge_attr)

            self.cache[_attr] = (edge_index, edge_attr)
        edge_index, edge_attr = self.cache[_attr]

        init_h = F.dropout(x, p=self.dropout, training=self.training)
        init_h = F.relu(self.fc_layers[0](init_h))

        h = init_h

        for layer in self.layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, edge_index, edge_attr, init_h)
            h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc_layers[1](h)
        return out

    def predict(self, data):
        return self.forward(data.x, data.edge_index)

    def get_optimizer(self, args):
        return torch.optim.Adam(
            [
                {"params": self.fc_parameters, "weight_decay": self.wd1},
                {"params": self.conv_parameters, "weight_decay": self.wd2},
            ],
            lr=args.lr,
        )
