import torch.nn as nn
import torch.nn.functional as F

from cogdl import experiment
from cogdl.layers import GCNLayer
from cogdl.models import BaseModel
from cogdl.datasets.planetoid_data import CoraDataset


class GCN(BaseModel):
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(in_feats, hidden_size)
        self.conv2 = GCNLayer(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        h = F.relu(self.conv1(graph, self.dropout(h)))
        h = self.conv2(graph, self.dropout(h))
        return h


if __name__ == "__main__":
    cora = CoraDataset()
    model = GCN(in_feats=cora.num_features, hidden_size=64, out_feats=cora.num_classes, dropout=0.1)
    experiment(dataset="cora", model=model, dw="node_classification_dw", mw="node_classification_mw")
