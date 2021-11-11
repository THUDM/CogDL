import torch
import torch.nn as nn

from cogdl.utils import edge_softmax


class DisenGCNLayer(nn.Module):
    """
    Implementation of `"Disentangled Graph Convolutional Networks" <http://proceedings.mlr.press/v97/ma19a.html>`_.
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

    def forward(self, graph, x):
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

        edge_index = graph.edge_index
        for _ in range(self.iterations):
            src_edge_attr = h_dst[edge_index[0]] * h_src[edge_index[1]]
            src_edge_attr = src_edge_attr.sum(dim=-1) / self.tau  # shape: (N, K)
            edge_attr_softmax = edge_softmax(graph, src_edge_attr).T  # shape: (E, K)
            edge_attr_softmax = edge_attr_softmax.unsqueeze(-1)  # shape: (K, E, 1)

            dst_edge_attr = h_src.index_select(0, edge_index[1]).permute(1, 0, 2)  # shape: (E, K, d) -> (K, E, d)
            dst_edge_attr = dst_edge_attr * edge_attr_softmax
            edge_index_ = edge_index[0].unsqueeze(-1).unsqueeze(0).repeat(self.K, 1, h.shape[-1])
            node_attr = torch.zeros(add_shape).to(device).scatter_add_(1, edge_index_, dst_edge_attr)  # (K, N, d)
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
