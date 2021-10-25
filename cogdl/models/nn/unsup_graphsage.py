import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel
from cogdl.layers import SAGELayer
from cogdl.models.nn.graphsage import sage_sampler
from cogdl.utils import RandomWalker


class SAGE(BaseModel):
    """
    Implementation of unsupervised GraphSAGE in paper `"Inductive Representation Learning on Large Graphs"` <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>

    Parameters
    ----------
    num_features : int
        Size of each input sample
    hidden_size : int
    num_layers : int
        The number of GNN layers.
    samples_size : list
        The number sampled neighbors of different orders
    dropout : float
    walk_length : int
        The length of random walk
    negative_samples : int
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[10, 10])
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            num_features=args.num_features,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            sample_size=args.sample_size,
            dropout=args.dropout,
        )

    def __init__(self, num_features, hidden_size, num_layers, sample_size, dropout):
        super(SAGE, self).__init__()
        self.adjlist = {}
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout

        shapes = [num_features] + [hidden_size] * num_layers
        self.convs = nn.ModuleList([SAGELayer(shapes[layer], shapes[layer + 1]) for layer in range(num_layers)])

    def forward(self, graph):
        x = graph.x
        for i in range(self.num_layers):
            edge_index_sp = self.sampling(graph.edge_index, self.sample_size[i]).to(x.device)
            with graph.local_graph():
                graph.edge_index = edge_index_sp
                x = self.convs[i](graph, x)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embed(self, data):
        emb = self.forward(data)
        return emb

    def sampling(self, edge_index, num_sample):
        return sage_sampler(self.adjlist, edge_index, num_sample)
