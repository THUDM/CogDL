import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

from cogdl import experiment
from cogdl.models import BaseModel
from cogdl.datasets.planetoid_data import CoraDataset


class GCN(BaseModel):
    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout):
        super(GCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList(
            [GCNConv(shapes[layer], shapes[layer + 1], cached=False) for layer in range(num_layers)]
        )

    def forward(self, graph):
        x = graph.x
        edge_index, edge_weight = torch.stack(graph.edge_index), graph.edge_weight
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    cora = CoraDataset()
    model = GCN(
        num_features=cora.num_features,
        hidden_size=64,
        num_classes=cora.num_classes,
        num_layers=2,
        dropout=0.5,
    )
    ret = experiment(dataset=cora, model=model)
