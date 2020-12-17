from typing import Any
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import MeanAggregator
from cogdl.trainers.sampled_trainer import NeighborSamplingTrainer

from .. import BaseModel, register_model


# edge index based sampler
# @profile
def sage_sampler(adjlist, edge_index, num_sample):
    # print(edge_index)
    if adjlist == {}:
        edge_index = edge_index.t().cpu().tolist()
        for i in edge_index:
            if not (i[0] in adjlist):
                adjlist[i[0]] = [i[1]]
            else:
                adjlist[i[0]].append(i[1])

    sample_list = []
    for i in adjlist:
        list = [[i, j] for j in adjlist[i]]
        if len(list) > num_sample:
            list = random.sample(list, num_sample)
        sample_list.extend(list)

    edge_idx = torch.LongTensor(sample_list).t()
    return edge_idx


class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphSAGELayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggr = MeanAggregator(in_feats, out_feats, cached=True)

    def forward(self, x, edge_index):
        adj_sp = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1]).float().to(x.device),
            (x.shape[0], x.shape[0]),
        ).to(x.device)
        x = self.aggr(x, adj_sp)
        return x


@register_model("graphsage")
class Graphsage(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, nargs='+', default=[128])
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[10, 10])
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--batch-size", type=int, default=128)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.sample_size,
            args.dropout,
        )

    def sampling(self, edge_index, num_sample):
        return sage_sampler(self.adjlist, edge_index, num_sample)

    def __init__(self, num_features, num_classes, hidden_size, num_layers, sample_size, dropout):
        super(Graphsage, self).__init__()
        assert num_layers == len(sample_size)
        self.adjlist = {}
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout
        shapes = [num_features] + hidden_size + [num_classes]
        self.convs = nn.ModuleList([GraphSAGELayer(shapes[layer], shapes[layer + 1]) for layer in range(num_layers)])

    def mini_forward(self, x, edge_index):
        for i in range(self.num_layers):
            edge_index_sp = self.sampling(edge_index, self.sample_size[i]).to(x.device)
            x = self.convs[i](x, edge_index_sp)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def mini_loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)

    def forward(self, *args):
        assert len(args) == 2
        if isinstance(args[1], torch.Tensor):
            return self.mini_forward(*args)
        else:
            x, adjs = args
            for i, (src_id, edge_index, size) in enumerate(adjs):
                edge_index = edge_index.to(self.device)
                output = self.convs[i](x, edge_index)
                x = output[0 : size[1]]
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return F.log_softmax(x, dim=-1)

    def node_classification_loss(self, *args):
        assert len(args) == 1 or len(args) == 3
        if len(args) == 1:
            return self.mini_loss(*args)
        else:
            x, adjs, y = args
            pred = self.forward(x, adjs)
            return F.nll_loss(pred, y)

    def inference(self, x_all, data_loader):
        for i in range(len(self.convs)):
            output = []
            for src_id, edge_index, size in data_loader:
                x = x_all[src_id].to(self.device)
                edge_index = edge_index.to(self.device)
                x = self.convs[i](x, edge_index)
                x = x[: size[1]]
                if i != self.num_layers - 1:
                    x = F.relu(x)
                output.append(x.cpu())
            x_all = torch.cat(output, dim=0)
        return F.log_softmax(x_all, dim=-1)

    @staticmethod
    def get_trainer(taskType: Any, args: Any):
        if args.dataset not in ["cora", "citeseer"]:
            return NeighborSamplingTrainer

    def set_data_device(self, device):
        self.device = device
