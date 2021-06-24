import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn.pool.topk_pool import filter_adj, topk

from cogdl.layers import GCNLayer
from cogdl.utils import split_dataset_general

from .. import BaseModel, register_model


class SAGPoolLayers(nn.Module):
    def __init__(self, nhid, ratio=0.8, Conv=GCNLayer, non_linearity=torch.tanh):
        super(SAGPoolLayers, self).__init__()
        self.nhid = nhid
        self.ratio = ratio
        self.score_layer = Conv(nhid, 1)
        self.non_linearity = non_linearity

    def forward(self, graph, x, batch=None):
        if batch is None:
            batch = graph.edge_index.new_zeros(x.size(0))
        score = self.score_layer(graph, x).squeeze()
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(graph.edge_index, graph.edge_weight, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm


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
        return split_dataset_general(dataset, args)

    def __init__(self, nfeat, nhid, nclass, dropout, pooling_ratio, pooling_layer_type):
        def __get_layer_from_str__(str):
            if str == "gcnconv":
                return GCNLayer
            return GCNLayer

        super(SAGPoolNetwork, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.pooling_ratio = pooling_ratio

        self.conv_layer_1 = GCNLayer(self.nfeat, self.nhid)
        self.conv_layer_2 = GCNLayer(self.nhid, self.nhid)
        self.conv_layer_3 = GCNLayer(self.nhid, self.nhid)

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

        with batch.local_graph():
            x = F.relu(self.conv_layer_1(batch, x))
            x, edge_index, _, batch_h, _ = self.pool_layer_1(batch, x, batch_h)
            out = torch.cat([gmp(x, batch_h), gap(x, batch_h)], dim=1)

            batch.edge_index = edge_index
            x = F.relu(self.conv_layer_2(batch, x))
            x, edge_index, _, batch_h, _ = self.pool_layer_2(batch, x, batch_h)
            out += torch.cat([gmp(x, batch_h), gap(x, batch_h)], dim=1)

            batch.edge_index = edge_index
            x = F.relu(self.conv_layer_3(batch, x))
            x, edge_index, _, batch_h, _ = self.pool_layer_3(batch, x, batch_h)
            out += torch.cat([gmp(x, batch_h), gap(x, batch_h)], dim=1)

        out = F.relu(self.lin_layer_1(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.lin_layer_2(out))
        return out
