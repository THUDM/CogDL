import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import ChebConv

from cogdl import experiment
from cogdl.models import BaseModel
from cogdl.datasets.planetoid_data import CoraDataset


class ChebyNet(BaseModel):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers, dropout, filter_size):
        super(ChebyNet, self).__init__()

        self.num_features = in_feats
        self.num_classes = out_feats
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.filter_size = filter_size
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.convs = nn.ModuleList(
            [ChebConv(shapes[layer], shapes[layer + 1], filter_size) for layer in range(num_layers)]
        )

    def forward(self, graph):
        x = graph.x
        edge_index = torch.stack(graph.edge_index)
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


if __name__ == "__main__":
    cora = CoraDataset()
    model = ChebyNet(
        in_feats=cora.num_features,
        hidden_size=64,
        out_feats=cora.num_classes,
        num_layers=2,
        dropout=0.5,
        filter_size=5,
    )
    ret = experiment(dataset=cora, model=model)
