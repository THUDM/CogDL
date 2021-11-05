import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GCNIILayer
from .. import BaseModel


class GCNII(BaseModel):
    """
    Implementation of GCNII in paper `"Simple and Deep Graph Convolutional Networks" <https://arxiv.org/abs/2007.02133>`_.

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
        parser.add_argument("--wd1", type=float, default=5e-4)
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
            actnn=args.actnn,
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
        actnn=False,
    ):
        super(GCNII, self).__init__()
        Layer = GCNIILayer
        Linear = nn.Linear
        Dropout = nn.Dropout
        ReLU = nn.ReLU(inplace=True)
        if actnn:
            try:
                from cogdl.layers.actgcnii_layer import ActGCNIILayer
                from actnn.layers import QLinear, QReLU, QDropout
            except Exception:
                print("Please install the actnn library first.")
                exit(1)
            Layer = ActGCNIILayer
            Linear = QLinear
            Dropout = QDropout
            ReLU = QReLU()

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(Linear(in_feats, hidden_size))
        self.fc_layers.append(Linear(hidden_size, out_feats))

        self.dropout = Dropout(dropout)
        self.activation = ReLU
        self.alpha = alpha
        self.lmbda = lmbda
        self.wd1 = wd1
        self.wd2 = wd2

        self.layers = nn.ModuleList(
            Layer(hidden_size, self.alpha, math.log(self.lmbda / (i + 1) + 1), residual) for i in range(num_layers)
        )

        self.fc_parameters = list(self.fc_layers.parameters())
        self.conv_parameters = list(self.layers.parameters())

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        init_h = self.dropout(x)
        init_h = self.activation(self.fc_layers[0](init_h))

        h = init_h

        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h, init_h)
            h = self.activation(h)
        h = self.dropout(h)
        out = self.fc_layers[1](h)
        return out

    def predict(self, graph):
        return self.forward(graph)

    def get_optimizer(self, args):
        return torch.optim.Adam(
            [
                {"params": self.fc_parameters, "weight_decay": self.wd1},
                {"params": self.conv_parameters, "weight_decay": self.wd2},
            ],
            lr=args.lr,
        )
