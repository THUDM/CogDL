import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .. import BaseModel, register_model
from cogdl.utils import mul_edge_softmax, spmm, mh_spmm, check_mh_spmm


class GATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, nhead=1, alpha=0.2, dropout=0.6, concat=True, residual=False):
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

        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        # nn.init.xavier_uniform_(self.a_l.data, gain=1.414)

    def forward(self, graph, x):
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0

        row, col = graph.edge_index
        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)[row]
        h_r = (self.a_r * h).sum(dim=-1)[col]
        edge_attention = self.leakyrelu(h_l + h_r)
        # edge_attention: E * H
        edge_attention = mul_edge_softmax(graph, edge_attention)
        edge_attention = self.dropout(edge_attention)

        if check_mh_spmm() and next(self.parameters()).device.type != "cpu":
            if self.nhead > 1:
                h_prime = mh_spmm(graph, edge_attention, h)
                out = h_prime.view(h_prime.shape[0], -1)
            else:
                edge_weight = edge_attention.view(-1)
                with graph.local_graph():
                    graph.edge_weight = edge_weight
                    out = spmm(graph, h.squeeze(1))
        else:
            with graph.local_graph():
                h_prime = []
                h = h.permute(1, 0, 2).contiguous()
                for i in range(self.nhead):
                    edge_weight = edge_attention[:, i]
                    graph.edge_weight = edge_weight
                    hidden = h[i]
                    assert not torch.isnan(hidden).any()
                    h_prime.append(spmm(graph, hidden))
            out = torch.cat(h_prime, dim=1)

        if self.residual:
            res = self.residual(x)
            out += res
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
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def forward(self, graph):
        x = graph.x
        for i, layer in enumerate(self.attentions):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = F.elu(x)
        return x

    def predict(self, graph):
        return self.forward(graph)
