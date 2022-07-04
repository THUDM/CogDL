import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.models import BaseModel
from cogdl.layers import MLP
from cogdl.layers import GINLayer
from cogdl.utils import split_dataset_general


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

    def split_dataset(cls, dataset, args):
        return split_dataset_general(dataset, args)

    def __init__(
        self,
        num_layers,
        in_feats,
        out_feats,
        hidden_dim,
        num_mlp_layers=1,
        eps=0,
        pooling="sum",
        train_eps=False,
        dropout=0.5,
    ):
        super(GIN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.num_layers = num_layers
        self.layer2 = nn.Linear(hidden_dim, out_feats)

        for i in range(num_layers - 1):
            if i == 0:
                self.mlp = MLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            else:
                self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            self.gin_layers.append(GINLayer(self.mlp, eps, train_eps))
            self.batch_norm.append(nn.BatchNorm1d(hidden_dim))

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, graph):
        h=graph.x
        for i in range(self.num_layers - 1):
            h = self.gin_layers[i](graph, h)
            h = self.batch_norm[i](h)
            h = F.relu(h)
            h = self.layer2(h)

        return F.log_softmax(h, dim=-1)
