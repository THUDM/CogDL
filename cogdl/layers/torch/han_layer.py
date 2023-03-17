import torch
import torch.nn as nn

from .gat_layer import GATLayer


class AttentionLayer(nn.Module):
    def __init__(self, num_features):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        att = self.linear(x).view(-1, 1, x.shape[1])
        return torch.matmul(att, x).squeeze(1)


class HANLayer(nn.Module):
    def __init__(self, num_edge, w_in, w_out):
        super(HANLayer, self).__init__()
        self.gat_layer = nn.ModuleList()
        for _ in range(num_edge):
            self.gat_layer.append(GATLayer(w_in, w_out // 8, 8))
        self.att_layer = AttentionLayer(w_out)

    def forward(self, graph, x):
        adj = graph.adj
        output = []
        with graph.local_graph():
            for i, edge in enumerate(adj):
                graph.edge_index = edge[0]
                output.append(self.gat_layer[i](graph, x))
        output = torch.stack(output, dim=1)

        return self.att_layer(output)
