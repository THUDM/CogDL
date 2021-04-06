import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, symmetric_normalization
from cogdl.trainers.m3s_trainer import M3STrainer
from .gcn import GraphConvolution


@register_model("m3s")
class M3S(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--num-clusters", type=int, default=50)
        parser.add_argument("--num-stages", type=int, default=10)
        parser.add_argument("--epochs-per-stage", type=int, default=50)
        parser.add_argument("--label-rate", type=float, default=1)
        parser.add_argument("--num-new-labels", type=int, default=2)
        parser.add_argument("--alpha", type=float, default=1)
        parser.add_argument("--approximate", action="store_true")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.dropout,
        )

    def __init__(self, num_features, hidden_size, num_classes, dropout):
        super(M3S, self).__init__()
        self.dropout = dropout
        self.gcn1 = GraphConvolution(num_features, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, num_classes)

    def get_embeddings(self, graph):
        graph.sym_norm()
        h = graph.x
        h = self.gcn1(graph, h)
        h = F.relu(F.dropout(h, self.dropout, training=self.training))
        return h.detach().cpu().numpy()

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        h = self.gcn1(graph, h)
        h = F.dropout(F.relu(h), self.dropout, training=self.training)
        h = self.gcn2(graph, h)
        return h

    def predict(self, data):
        return self.forward(data)

    @staticmethod
    def get_trainer(taskType, args):
        return M3STrainer
