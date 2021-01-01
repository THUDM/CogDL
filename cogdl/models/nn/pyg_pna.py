import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding, Sequential, ReLU, Linear
from .. import BaseModel, register_model
from cogdl.data import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool


@register_model("pyg_pna")
class PNA(BaseModel):
    r"""Implements a single convolutional layer of the Principal Neighbourhood Aggregation Networks 
    in paper `"Principal Neighbourhood Aggregation for Graph Nets" <https://arxiv.org/abs/2004.05718>.`
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num_features", type=int)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--hidden_size", type=int, default=60)
        parser.add_argument("--avg_deg", type=int, default=1)
        parser.add_argument("--layer", type=int, default=4)
        parser.add_argument("--pre_layers", type=int, default=1)
        parser.add_argument("--towers", type=int, default=5)
        parser.add_argument("--post_layers", type=int, default=1)
        parser.add_argument("--edge_dim", type=int, default=None)
        parser.add_argument("--aggregators", type=str, nargs="+", default=['mean', 'min', 'max', 'std'])
        parser.add_argument("--scalers", type=str, nargs="+", default=['identity', 'amplification', 'attenuation'])
        parser.add_argument("--divide_input", action='store_true', default=False)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.avg_deg,
            args.layer,
            args.pre_layers,
            args.towers,
            args.post_layers,
            args.edge_dim,
            args.aggregators,
            args.scalers,
            args.divide_input
        )

    @classmethod
    def split_dataset(cls, dataset, args):
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

    def __init__(self, num_feature, num_classes, hidden_size, avg_deg,
                 layer=4, pre_layers=1, towers=5, post_layers=1,
                 edge_dim=None,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 divide_input=False):
        super(PNA, self).__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim

        avg_deg = torch.tensor(avg_deg)
        emd_side = self.hidden_size // num_feature
        self.hidden_size = emd_side * num_feature
        self.node_emb = Embedding(num_feature, emd_side)
        if self.edge_dim is not None:
            self.edge_emb = Embedding(4, edge_dim)        

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(layer):
            conv = PNAConv(in_channels=self.hidden_size, out_channels=self.hidden_size,
                           aggregators=aggregators, scalers=scalers, deg=avg_deg,
                           edge_dim=edge_dim, towers=towers, pre_layers=pre_layers, post_layers=post_layers,
                           divide_input=divide_input)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_size))

        self.mlp = Sequential(Linear(self.hidden_size, self.hidden_size // 2), 
                              ReLU(), 
                              Linear(self.hidden_size // 2, self.hidden_size // 4), 
                              ReLU(),
                              Linear(self.hidden_size // 4, num_classes))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, b):
        x = b.x
        edge_index = b.edge_index
        edge_attr = b.edge_attr
        batch_h = b.batch
        n = x.shape[0]

        x = self.node_emb(x.long())
        x = x.reshape([n, -1])

        if self.edge_dim is None:
            edge_attr = None
        else:
            edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        x = global_add_pool(x, batch_h)
        out = self.mlp(x)

        if b.y is not None:
            return out, self.criterion(out, b.y)
        return out, None
