import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

from .. import BaseModel, register_model
from .pyg_gin import GINMLP
from cogdl.data import DataLoader, Data


@register_model("dgcnn")
class DGCNN(BaseModel):
    r"""EdgeConv and DynamicGraph in paper `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/pdf/1801.07829.pdf>__ .`

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    out_feats : int
        Size of each output sample.
    hidden_dim : int
        Dimension of hidden layer embedding.
    k : int
        Number of neareast neighbors.
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=0.001)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        if "ModelNet" in args.dataset:
            train_data = [Data(x=d.pos, y=d.y) for d in dataset["train"]]
            test_data = [Data(x=d.pos, y=d.y) for d in dataset["test"]]
            train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=6)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=6, shuffle=False)
            return train_loader, test_loader, test_loader
        else:
            random.shuffle(dataset)
            train_size = int(len(dataset) * args.train_ratio)
            test_size = int(len(dataset) * args.test_ratio)
            bs = args.batch_size
            train_loader = DataLoader(dataset[:train_size], batch_size=bs)
            test_loader = DataLoader(dataset[-test_size:], batch_size=bs)
            if args.train_ratio + args.test_ratio < 1:
                valid_loader = DataLoader(dataset[train_size:-test_size], batch_size=bs)
            else:
                valid_loader = test_loader
            return train_loader, valid_loader, test_loader

    def __init__(self, in_feats, hidden_dim, out_feats, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        mlp1 = nn.Sequential(GINMLP(2*in_feats, hidden_dim, hidden_dim, num_layers=3), nn.ReLU(), nn.BatchNorm1d(hidden_dim))
        mlp2 = nn.Sequential(GINMLP(2*hidden_dim, 2*hidden_dim, 2*hidden_dim, num_layers=1), nn.ReLU(), nn.BatchNorm1d(2*hidden_dim))
        self.conv1 = DynamicEdgeConv(mlp1, k, "max")
        self.conv2 = DynamicEdgeConv(mlp2, k, "max")
        self.linear = nn.Linear(hidden_dim + 2*hidden_dim, 1024)
        self.final_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, out_feats)
        )

    def forward(self, batch):
        h = batch.x
        h1 = self.conv1(h, batch.batch)
        h2 = self.conv2(h1, batch.batch)
        h = self.linear(torch.cat([h1, h2], dim=1))
        h = global_max_pool(h, batch.batch)
        out = self.final_mlp(h)
        out = F.log_softmax(out, dim=-1)
        if batch.y is not None:
            loss = F.nll_loss(out, batch.y)
            return out, loss
        return out, None

