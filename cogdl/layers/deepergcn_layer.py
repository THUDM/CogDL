import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.utils import get_activation, mul_edge_softmax
from torch.utils.checkpoint import checkpoint


class GENConv(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        aggr="softmax_sg",
        beta=1.0,
        p=1.0,
        learn_beta=False,
        learn_p=False,
        use_msg_norm=False,
        learn_msg_scale=True,
    ):
        super(GENConv, self).__init__()
        self.use_msg_norm = use_msg_norm
        self.mlp = nn.Linear(in_feat, out_feat)

        self.message_encoder = torch.nn.ReLU()

        self.aggr = aggr
        if aggr == "softmax_sg":
            self.beta = torch.nn.Parameter(
                torch.Tensor(
                    [
                        beta,
                    ]
                ),
                requires_grad=learn_beta,
            )
        else:
            self.register_buffer("beta", None)
        if aggr == "powermean":
            self.p = torch.nn.Parameter(
                torch.Tensor(
                    [
                        p,
                    ]
                ),
                requires_grad=learn_p,
            )
        else:
            self.register_buffer("p", None)
        self.eps = 1e-7

        self.s = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=learn_msg_scale)
        self.act = nn.ReLU()

    def message_norm(self, x, msg):
        x_norm = torch.norm(x, dim=1, p=2)
        msg_norm = F.normalize(msg, p=2, dim=1)
        msg_norm = msg_norm * x_norm.unsqueeze(-1)
        return x + self.s * msg_norm

    def forward(self, graph, x):
        edge_index = graph.edge_index
        dim = x.shape[1]
        edge_msg = x[edge_index[1]]  # if edge_attr is None else x[edge_index[1]] + edge_attr
        edge_msg = self.act(edge_msg) + self.eps

        if self.aggr == "softmax_sg":
            h = mul_edge_softmax(graph, self.beta * edge_msg)
            h = edge_msg * h
        elif self.aggr == "softmax":
            h = mul_edge_softmax(graph, edge_msg)
            h = edge_msg * h
        elif self.aggr == "powermean":
            deg = graph.degrees()
            h = edge_msg.pow(self.t) / deg[edge_index[0]].unsqueeze(-1)
        else:
            raise NotImplementedError

        h = torch.zeros_like(x).scatter_add_(dim=0, index=edge_index[0].unsqueeze(-1).repeat(1, dim), src=h)
        if self.aggr == "powermean":
            h = h.pow(1.0 / self.p)
        if self.use_msg_norm:
            h = self.message_norm(x, h)
        h = self.mlp(h)
        return h


class DeepGCNLayer(nn.Module):
    """
    Implementation of DeeperGCN in paper `"DeeperGCN: All You Need to Train Deeper GCNs"` <https://arxiv.org/abs/2006.07739>

    Parameters
    -----------
    in_feat : int
        Size of each input sample
    out_feat : int
        Size of each output sample
    conv : class
        Base convolution layer.
    connection : str
        Residual connection type, `res` or `res+`.
    activation : str
    dropout : float
    checkpoint_grad : bool
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        conv,
        connection="res",
        activation="relu",
        dropout=0.0,
        checkpoint_grad=False,
    ):
        super(DeepGCNLayer, self).__init__()
        self.conv = conv
        self.activation = get_activation(activation)
        self.dropout = dropout
        self.connection = connection
        self.norm = nn.BatchNorm1d(out_feat, affine=True)
        self.checkpoint_grad = checkpoint_grad

    def forward(self, graph, x):
        if self.connection == "res+":
            h = self.norm(x)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.checkpoint_grad:
                h = checkpoint(self.conv, graph, h)
            else:
                h = self.conv(graph, h)
        elif self.connection == "res":
            h = self.conv(graph, x)
            h = self.norm(h)
            h = self.activation(h)
        else:
            raise NotImplementedError
        return x + h
