import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from .. import BaseModel, register_model
from cogdl.data import DataLoader


class SAGPoolLayers(nn.Module):
    def __init__(self, nhid, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPoolLayers, self).__init__()
        self.nhid = nhid
        self.ratio = ratio
        self.score_layer = Conv(nhid, 1)
        self.non_linearity = non_linearity

    def forward(self, input, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(input.size(0))
        score = self.score_layer(input, edge_index).squeeze()
        perm = topk(score, self.ratio, batch)
        input = input[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return input, edge_index, edge_attr, batch, perm


@register_model("sagpool")
class SAGPoolNetwork(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--pooling-ratio", type=float, default=0.5)
        parser.add_argument("--pooling-layer-type", type=str, default="gcnconv")
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.dropout,
            args.pooling_ratio,
            args.pooling_layer_type,
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

    def __init__(self, nfeat, nhid, nclass, dropout, pooling_ratio, pooling_layer_type):
        def __get_layer_from_str__(str):
            if str == "gcnconv":
                return GCNConv
            return GCNConv

        super(SAGPoolNetwork, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.pooling_ratio = pooling_ratio

        self.conv_layer_1 = GCNConv(self.nfeat, self.nhid)
        self.conv_layer_2 = GCNConv(self.nhid, self.nhid)
        self.conv_layer_3 = GCNConv(self.nhid, self.nhid)

        self.pool_layer_1 = SAGPoolLayers(
            self.nhid, Conv=__get_layer_from_str__(pooling_layer_type), ratio=self.pooling_ratio
        )
        self.pool_layer_2 = SAGPoolLayers(
            self.nhid, Conv=__get_layer_from_str__(pooling_layer_type), ratio=self.pooling_ratio
        )
        self.pool_layer_3 = SAGPoolLayers(
            self.nhid, Conv=__get_layer_from_str__(pooling_layer_type), ratio=self.pooling_ratio
        )

        self.lin_layer_1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin_layer_2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin_layer_3 = torch.nn.Linear(self.nhid // 2, self.nclass)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_h = batch.batch

        x = F.relu(self.conv_layer_1(x, edge_index))
        x, edge_index, _, batch_h, _ = self.pool_layer_1(x, edge_index, None, batch_h)
        out = torch.cat([gmp(x, batch_h), gap(x, batch_h)], dim=1)

        x = F.relu(self.conv_layer_2(x, edge_index))
        x, edge_index, _, batch_h, _ = self.pool_layer_2(x, edge_index, None, batch_h)
        out += torch.cat([gmp(x, batch_h), gap(x, batch_h)], dim=1)

        x = F.relu(self.conv_layer_3(x, edge_index))
        x, edge_index, _, batch_h, _ = self.pool_layer_3(x, edge_index, None, batch_h)
        out += torch.cat([gmp(x, batch_h), gap(x, batch_h)], dim=1)

        out = F.relu(self.lin_layer_1(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.lin_layer_2(out))
        out = F.log_softmax(self.lin_layer_3(out), dim=-1)
        if batch.y is not None:
            loss = F.nll_loss(out, batch.y)
            return out, loss
        return out, None
