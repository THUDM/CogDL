import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import register_model, BaseModel
from cogdl.utils import mul_edge_softmax, remove_self_loops


class DisenGCNLayer(nn.Module):
    """
        Implementation of "Disentangled Graph Convolutional Networks" <http://proceedings.mlr.press/v97/ma19a.html>.
    """
    def __init__(self, in_feats, out_feats, K, iterations, tau=1.0, activation="leaky_relu"):
        super(DisenGCNLayer, self).__init__()
        self.K = K
        self.tau = tau
        self.iterations = iterations
        self.factor_dim = int(out_feats / K)

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.zeros_(self.bias.data)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        device = x.device

        h = self.activation(torch.matmul(x, self.weight) + self.bias)

        h = h.split(self.factor_dim, dim=-1)
        h = torch.cat([dt.unsqueeze(0) for dt in h], dim=0)
        norm = h.pow(2).sum(dim=-1).sqrt().unsqueeze(-1)

        # multi-channel softmax: faster
        h_normed = h / norm  # (K, N, d)
        h_src = h_dst = h_normed.permute(1, 0, 2)  # (N, K, d)
        add_shape = h.shape  # (K, N, d)

        for _ in range(self.iterations):
            src_edge_attr = h_dst[edge_index[0]] * h_src[edge_index[1]]
            src_edge_attr = src_edge_attr.sum(dim=-1) / self.tau  # shape: (N, K)
            edge_attr_softmax = mul_edge_softmax(edge_index, src_edge_attr, shape=(num_nodes, num_nodes)) # shape: (E, K)
            edge_attr_softmax = edge_attr_softmax.t().unsqueeze(-1)  # shape: (K, E, 1)

            dst_edge_attr = h_src.index_select(0, edge_index[1]).permute(1, 0, 2)  # shape: (E, K, d) -> (K, E, d)
            dst_edge_attr = dst_edge_attr * edge_attr_softmax
            edge_index_ = edge_index[0].unsqueeze(-1).unsqueeze(0).repeat(self.K, 1, h.shape[-1])
            node_attr = torch.zeros(add_shape).to(device).scatter_add_(1, edge_index_, dst_edge_attr) # (K, N, d)
            node_attr = node_attr + h_normed
            node_attr_norm = node_attr.pow(2).sum(-1).sqrt().unsqueeze(-1)  # shape: (K, N, 1)
            node_attr = (node_attr / node_attr_norm).permute(1, 0, 2)  # shape: (N, K, d)
            h_dst = node_attr

        h_dst = h_dst.reshape(num_nodes, -1)

        # Calculate the softmax of each channel separately
        # h_src = h_dst = h / norm  # (K, N, d)
        #
        # for _ in range(self.iterations):
        #     for i in range(self.K):
        #         h_attr = h_dst[i]
        #         edge_attr = h_attr[edge_index[0]] * h_src[i][edge_index[1]]
        #
        #         edge_attr = edge_attr.sum(-1)/self.tau
        #         edge_attr = edge_softmax(edge_index, edge_attr, shape=(num_nodes, num_nodes))
        #
        #         node_attr = spmm(edge_index, edge_attr, h_src[i])
        #
        #         node_attr = node_attr + h_src[i]
        #         h_src[i] = node_attr / node_attr.pow(2).sum(-1).sqrt().unsqueeze(-1)
        #
        # h_dst = h_dst.permute(1, 0, 2).reshape(num_nodes, -1)

        return h_dst


@register_model("disengcn")
class DisenGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--K", type=int, nargs="+", default=[16, 8])
        parser.add_argument("--iterations", type=int, default=7)
        parser.add_argument("--tau", type=float, default=1)
        parser.add_argument("--activation", type=str, default="leaky_relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feats=args.num_features,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes,
            K=args.K,
            iterations=args.iterations,
            tau=args.tau,
            dropout=args.dropout,
            activation=args.activation,
        )

    def __init__(self, in_feats, hidden_size, num_classes, K, iterations, tau, dropout, activation):
        super(DisenGCN, self).__init__()
        self.K = K
        self.iterations = iterations
        self.dropout = dropout
        self.activation = activation
        self.num_layers = len(K)

        self.weight = nn.Parameter(torch.Tensor(hidden_size, num_classes))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.reset_parameters()

        shapes = [in_feats] + [hidden_size] * self.num_layers
        self.layers = nn.ModuleList(
            DisenGCNLayer(shapes[i], shapes[i + 1], K[i], iterations, tau, activation)
            for i in range(self.num_layers)
        )

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.zeros_(self.bias.data)

    def forward(self, x, edge_index):
        h = x
        edge_index, _ = remove_self_loops(edge_index)
        for layer in self.layers:
            h = layer(h, edge_index)
            # h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = torch.matmul(h, self.weight) + self.bias
        return F.log_softmax(out, dim=-1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask]
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
