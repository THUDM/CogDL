import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

from .. import BaseModel, register_model
from .gin import GINMLP
from cogdl.data import DataLoader, Data


@register_model("dgcnn")
class DGCNN(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--batch-size", type=int, default=32)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_classes,
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        train_data = [Data(x=d.pos, y=d.y) for d in dataset["train"]]
        test_data = [Data(x=d.pos, y=d.y) for d in dataset["test"]]
        train_loader = DataLoader(train_data, batch_size=args.batch_size)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        return train_loader, test_loader, test_loader

    def __init__(self, out_feats, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        self.conv1 = DynamicEdgeConv(GINMLP(2*3, 64, 64, num_layers=4), k, "max")
        self.conv2 = DynamicEdgeConv(GINMLP(2*64, 128, 128, num_layers=1), k, "max")
        self.linear = nn.Linear(128+64, 1024)
        self.final_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, out_feats)
        )

    def forward(self, x, edge_index=None, batch=None, label=None):
        h = x
        h1 = self.conv1(h)
        h2 = self.conv2(h)
        h = self.linear(torch.cat([h1, h2], dim=1))
        h = global_max_pool(h, batch)
        out = self.final_mlp(h)
        if label is not None:
            loss = F.nll_loss(F.log_softmax(out), label)
            return out, loss
        return out, None

