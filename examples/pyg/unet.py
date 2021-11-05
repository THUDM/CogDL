import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj

from cogdl import experiment
from cogdl.models import BaseModel
from cogdl.datasets.planetoid_data import CoraDataset


class UNet(BaseModel):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers, dropout, num_nodes):
        super(UNet, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.unet = GraphUNet(
            self.in_feats, self.hidden_size, self.out_feats, depth=3, pool_ratios=[2000 / num_nodes, 0.5], act=F.elu
        )

    def forward(self, graph):
        x = graph.x
        edge_index = torch.stack(graph.edge_index)
        edge_index, _ = dropout_adj(
            edge_index, p=0.2, force_undirected=True, num_nodes=x.shape[0], training=self.training
        )
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.unet(x, edge_index)
        return x


if __name__ == "__main__":
    cora = CoraDataset()
    model = UNet(
        in_feats=cora.num_features,
        hidden_size=64,
        out_feats=cora.num_classes,
        num_layers=2,
        dropout=0.1,
        num_nodes=cora.num_nodes,
    )
    ret = experiment(dataset=cora, model=model)
