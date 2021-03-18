import torch
import torch.nn as nn
from cogdl.utils import spmm


class MixHopLayer(nn.Module):
    def __init__(self, num_features, adj_pows, dim_per_pow):
        super(MixHopLayer, self).__init__()
        self.num_features = num_features
        self.adj_pows = adj_pows
        self.dim_per_pow = dim_per_pow
        self.total_dim = 0
        self.linears = torch.nn.ModuleList()
        for dim in dim_per_pow:
            self.linears.append(nn.Linear(num_features, dim))
            self.total_dim += dim
        # self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

    def adj_pow_x(self, graph, x, p):
        for _ in range(p):
            x = spmm(graph, x)
        return x

    def forward(self, graph, x):
        graph.sym_norm()
        output_list = []
        for p, linear in zip(self.adj_pows, self.linears):
            output = linear(self.adj_pow_x(graph, x, p))
            output_list.append(output)

        return torch.cat(output_list, dim=1)
