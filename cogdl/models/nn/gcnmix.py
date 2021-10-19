import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import spmm
from .. import BaseModel


def mix_hidden_state(feat, target, train_index, alpha):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1
    permuted_index = train_index[torch.randperm(train_index.size(0))]
    feat[train_index] = lamb * feat[train_index] + (1 - lamb) * feat[permuted_index]
    return feat, target[train_index], target[permuted_index], lamb


class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNConv, self).__init__()
        self.weight = nn.Linear(in_features=in_feats, out_features=out_feats)
        self.edge_index = None
        self.edge_attr = None

    def forward(self, graph, x):
        h = self.weight(x)
        h = spmm(graph, h)
        return h

    def forward_aux(self, x):
        return self.weight(x)


class GCNMix(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--k", type=int, default=10)
        parser.add_argument("--temperature", type=float, default=0.1)
        # parser.add_argument("--rampup-starts", type=int, default=500)
        # parser.add_argument("--rampup_ends", type=int, default=1000)
        # parser.add_argument("--mixup-consistency", type=float, default=10.0)
        # parser.add_argument("--ema-decay", type=float, default=0.999)
        # parser.add_argument("--tau", type=float, default=1.0)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feat=args.num_features,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes,
            k=args.k,
            temperature=args.temperature,
            alpha=args.alpha,
            dropout=args.dropout,
        )

    def __init__(self, in_feat, hidden_size, num_classes, k, temperature, alpha, dropout):
        super(GCNMix, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.k = k
        self.temperature = temperature

        self.input_gnn = GCNConv(in_feat, hidden_size)
        self.hidden_gnn = GCNConv(hidden_size, num_classes)
        self.loss_f = nn.BCELoss()

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.input_gnn(graph, h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.hidden_gnn(graph, h)
        return h

    def forward_aux(self, x, label, train_index, mix_hidden=True, layer_mix=1):
        h = F.dropout(x, p=self.dropout, training=self.training)
        assert layer_mix in (0, 1)
        if layer_mix == 0:
            h, target, target_mix, lamb = mix_hidden_state(h, label, train_index, self.alpha)
        h = self.input_gnn.forward_aux(h)
        h = F.relu(h)
        if layer_mix == 1:
            h, target, target_mix, lamb = mix_hidden_state(h, label, train_index, self.alpha)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.hidden_gnn.forward_aux(h)
        target_label = lamb * target + (1 - lamb) * target_mix
        return h, target_label

    def predict_noise(self, data, tau=1):
        out = self.forward(data) / tau
        return out
