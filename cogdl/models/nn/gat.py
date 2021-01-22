import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, mul_edge_softmax, spmm


class GATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, nhead=1, alpha=0.2, dropout=0.6, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.nhead = nhead

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameteres()

    def reset_parameteres(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.W)

    def forward(self, x, edge):
        N = x.size()[0]
        # h = self.W(x).view(-1, self.nhead, self.out_features)
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        # h: N * H * d
        if torch.isnan(h).any():
            print("NaN in Graph Attention")
            h[torch.isnan(h)] = 0

        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)[edge[0, :]]
        h_r = (self.a_r * h).sum(dim=-1)[edge[1, :]]
        edge_attention = self.leakyrelu(h_l + h_r)
        # edge_e: E * H
        edge_attention = mul_edge_softmax(edge, edge_attention, shape=(N, N))

        edge_attention = edge_attention.view(-1)
        edge_attention = self.dropout(edge_attention)

        num_edges = edge.shape[1]
        num_nodes = x.shape[0]
        edge_index = edge.view(-1)
        edge_index = edge_index.unsqueeze(0).repeat(self.nhead, 1)
        add_num = torch.arange(0, self.nhead * num_nodes, num_nodes).view(-1, 1).to(edge_index.device)
        edge_index = edge_index + add_num
        edge_index = edge_index.split((num_edges, num_edges), dim=1)

        row, col = edge_index
        row = row.reshape(-1)
        col = col.reshape(-1)
        edge_index = torch.stack([row, col])

        h_prime = spmm(edge_index, edge_attention, h.permute(1, 0, 2).reshape(num_nodes * self.nhead, -1))
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.split([num_nodes] * self.nhead)

        if self.concat:
            # if this layer is not last layer,
            out = torch.cat(h_prime, dim=1)
        else:
            # if this layer is last layer,
            out = sum(h_prime) / self.nhead
        return out

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


@register_model("gat")
class GAT(BaseModel):
    r"""The GAT model from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
        alpha (float) : Coefficient of leaky_relu.
        nheads (int) : Number of attention heads.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--nheads", type=int, default=8)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.dropout,
            args.alpha,
            args.nheads,
        )

    def __init__(self, in_feats, hidden_size, out_features, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attention = GATLayer(in_feats, hidden_size, dropout=dropout, alpha=alpha, nhead=nheads, concat=True)

        self.out_att = GATLayer(hidden_size * nheads, out_features, dropout=dropout, alpha=alpha, nhead=1, concat=False)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.attention(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
