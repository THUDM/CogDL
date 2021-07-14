import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import register_model, BaseModel
from cogdl.layers import SAGELayer
from cogdl.models.nn.graphsage import sage_sampler
from cogdl.trainers.self_supervised_trainer import SelfSupervisedPretrainer
from cogdl.utils import RandomWalker


@register_model("unsup_graphsage")
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
        parser.add_argument("--walk-length", type=int, default=10)
        parser.add_argument("--negative-samples", type=int, default=30)
        parser.add_argument("--lr", type=float, default=0.001)

        parser.add_argument("--max-epochs", type=int, default=3000)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            num_features=args.num_features,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            sample_size=args.sample_size,
            dropout=args.dropout,
            walk_length=args.walk_length,
            negative_samples=args.negative_samples,
        )

    def __init__(self, num_features, hidden_size, num_layers, sample_size, dropout, walk_length, negative_samples):
        super(SAGE, self).__init__()
        self.adjlist = {}
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout
        self.walk_length = walk_length
        self.num_negative_samples = negative_samples
        self.walk_res = None
        self.num_nodes = 0
        self.negative_samples = 1

        shapes = [num_features] + [hidden_size] * num_layers

        self.convs = nn.ModuleList([SAGELayer(shapes[layer], shapes[layer + 1]) for layer in range(num_layers)])
        self.random_walker = RandomWalker()

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

    def node_classification_loss(self, data):
        return self.loss(data)

    def self_supervised_loss(self, data):
        return self.loss(data)

    def loss(self, data):
        x = self.forward(data)
        device = x.device

        self.random_walker.build_up(data.edge_index, data.x.shape[0])
        walk_res = self.random_walker.walk(
            start=torch.arange(0, x.shape[0]).to(device), walk_length=self.walk_length + 1
        )
        self.walk_res = torch.as_tensor(walk_res)[:, 1:]

        if not self.num_nodes:
            self.num_nodes = max(data.edge_index[0].max(), data.edge_index[1].max()).item() + 1

        # if self.negative_samples is None:
        self.negative_samples = torch.from_numpy(
            np.random.choice(self.num_nodes, (self.num_nodes, self.num_negative_samples))
        ).to(device)

        pos_loss = -torch.log(
            torch.sigmoid(torch.sum(x.unsqueeze(1).repeat(1, self.walk_length, 1) * x[self.walk_res], dim=-1))
        ).mean()
        neg_loss = -torch.log(
            torch.sigmoid(
                -torch.sum(x.unsqueeze(1).repeat(1, self.num_negative_samples, 1) * x[self.negative_samples], dim=-1)
            )
        ).mean()
        return (pos_loss + neg_loss) / 2

    def embed(self, data):
        emb = self.forward(data)
        return emb

    def sampling(self, edge_index, num_sample):
        return sage_sampler(self.adjlist, edge_index, num_sample)

    @staticmethod
    def get_trainer(args):
        return SelfSupervisedPretrainer
