import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv

from cogdl import experiment
from cogdl.models import BaseModel
from cogdl.datasets.planetoid_data import CoraDataset


class GAT(BaseModel):
    def __init__(self, in_feats, hidden_size, out_feats, num_heads, dropout):
        super(GAT, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.conv1 = GATConv(in_feats, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads, out_feats, dropout=dropout)

    def forward(self, graph):
        x = graph.x
        edge_index = torch.stack(graph.edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return x


if __name__ == "__main__":
    cora = CoraDataset()
    model = GAT(in_feats=cora.num_features, hidden_size=64, out_feats=cora.num_classes, num_heads=2, dropout=0.1)
    ret = experiment(dataset=cora, model=model, dw="node_classification_dw", mw="node_classification_mw")
