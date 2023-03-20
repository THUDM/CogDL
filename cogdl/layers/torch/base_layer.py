import torch
import torch.nn as nn


class BaseLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, graph, x):
        m = self.message(x[graph.edge_index[0]])
        return self.aggregate(graph, m)

    def message(self, x):
        return x

    def aggregate(self, graph, x):
        result = torch.zeros(graph.num_nodes, x.shape[1], dtype=x.dtype).to(x.device)
        result.scatter_add_(0, graph.edge_index[1].unsqueeze(1).expand(-1, x.shape[1]), x)
        return result
