import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

from cogdl import experiment
from cogdl.models import BaseModel
from cogdl.models.nn.mlp import MLP
from cogdl.utils import split_dataset_general
from cogdl.datasets.tu_data import MUTAGDataset


class DGCNN(BaseModel):
    r"""EdgeConv and DynamicGraph in paper `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/pdf/1801.07829.pdf>__ .`
    """

    @classmethod
    def split_dataset(cls, dataset, args):
        return split_dataset_general(dataset, args)

    def __init__(self, in_feats, hidden_size, out_feats, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        mlp1 = nn.Sequential(
            MLP(2 * in_feats, hidden_size, hidden_size, num_layers=3, norm="batchnorm"),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        mlp2 = nn.Sequential(
            MLP(2 * hidden_size, 2 * hidden_size, 2 * hidden_size, num_layers=1, norm="batchnorm"),
            nn.ReLU(),
            nn.BatchNorm1d(2 * hidden_size),
        )
        self.conv1 = DynamicEdgeConv(mlp1, k, "max")
        self.conv2 = DynamicEdgeConv(mlp2, k, "max")
        self.linear = nn.Linear(hidden_size + 2 * hidden_size, 1024)
        self.final_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, out_feats),
        )

    def forward(self, batch):
        h = batch.x
        h1 = self.conv1(h, batch.batch)
        h2 = self.conv2(h1, batch.batch)
        h = self.linear(torch.cat([h1, h2], dim=1))
        h = global_max_pool(h, batch.batch)
        out = self.final_mlp(h)
        return out


if __name__ == "__main__":
    mutag = MUTAGDataset()
    model = DGCNN(
        in_feats=mutag.num_features,
        hidden_size=64,
        out_feats=mutag.num_classes,
        k=20,
        dropout=0.5,
    )
    ret = experiment(dataset=mutag, model=model, dw="graph_classification_dw", mw="graph_classification_mw")
