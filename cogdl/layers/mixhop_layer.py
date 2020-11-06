import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

    def adj_pow_x(self, x, adj, p):
        for _ in range(p):
            x = torch.spmm(adj, x)
        return x

    def forward(self, x, edge_index):
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1]).float(),
            (x.shape[0], x.shape[0]),
            device=x.device
        )
        output_list = []
        for p, linear in zip(self.adj_pows, self.linears):
            output = linear(self.adj_pow_x(x, adj, p))
            output_list.append(output)
        
        return torch.cat(output_list, dim=1)

if __name__ == "__main__":
    layer = MixHopLayer(10, [1, 3], [16, 32])
    x = torch.ones(5, 10)
    adj = torch.LongTensor([[0, 1, 1, 2, 2, 3, 4], [1, 2, 3, 0, 4, 4, 1]])
    output = layer(x, adj)
    print(output.shape)
