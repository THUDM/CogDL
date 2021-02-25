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

    def __init__(
        self, in_features, out_features, nhead=1, alpha=0.2, dropout=0.6, concat=True, residual=False, fast_mode=False
    ):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.nhead = nhead
        self.fast_mode = fast_mode

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if residual:
            out_features = out_features * nhead if concat else out_features
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.register_buffer("residual", None)
        self.reset_parameters()

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.W)

    def forward(self, x, edge):
        N = x.size()[0]
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        # h: N * H * d
        if torch.isnan(self.W.data).any():
            # print("NaN in Graph Attention, ", self.nhead)
            h[torch.isnan(h)] = 0

        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)[edge[0, :]]
        h_r = (self.a_r * h).sum(dim=-1)[edge[1, :]]
        edge_attention = self.leakyrelu(h_l + h_r)
        # edge_e: E * H
        edge_attention = mul_edge_softmax(edge, edge_attention, shape=(N, N))

        num_edges = edge.shape[1]
        num_nodes = x.shape[0]

        if self.fast_mode:
            edge_attention = edge_attention.view(-1)
            edge_attention = self.dropout(edge_attention)

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
        else:
            h_prime = []
            h = h.permute(1, 0, 2).contiguous()
            for i in range(self.nhead):
                edge_weight = edge_attention[:, i]
                hidden = h[i]
                assert not torch.isnan(hidden).any()
                h_prime.append(spmm(edge, edge_weight, hidden))

        if self.residual:
            res = self.residual(x)
        else:
            res = 0

        if self.concat:
            # if this layer is not last layer,
            out = torch.cat(h_prime, dim=1) + res
        else:
            # if this layer is last layer,
            out = sum(h_prime) / self.nhead + res
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
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--last-nhead", type=int, default=1)
        parser.add_argument("--fast-mode", action="store_true", default=False)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.alpha,
            args.nhead,
            args.residual,
            args.last_nhead,
            args.fast_mode,
        )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_features,
        num_layers,
        dropout,
        alpha,
        nhead,
        residual,
        last_nhead,
        fast_mode=False,
    ):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.attentions.append(
            GATLayer(
                in_feats,
                hidden_size,
                nhead=nhead,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                residual=residual,
                fast_mode=fast_mode,
            )
        )
        for i in range(num_layers - 2):
            self.attentions.append(
                GATLayer(
                    hidden_size * nhead,
                    hidden_size,
                    nhead=nhead,
                    dropout=dropout,
                    alpha=alpha,
                    concat=True,
                    residual=residual,
                    fast_mode=fast_mode,
                )
            )
        self.attentions.append(
            GATLayer(
                hidden_size * nhead,
                out_features,
                dropout=dropout,
                alpha=alpha,
                concat=False,
                nhead=last_nhead,
                residual=False,
                fast_mode=fast_mode,
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)

        for i, layer in enumerate(self.attentions):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
