import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import MeanAggregator

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
    # for i in edge_index
    # print("sampled",edge_index)
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
            torch.ones(edge_index.shape[1]).float(),
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
        parser.add_argument("--sample-size",type=int,nargs='+', default=[10, 10])
        parser.add_argument("--dropout", type=float, default=0.5)
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

    def __init__(
        self, num_features, num_classes, hidden_size, num_layers, sample_size, dropout
    ):
        super(Graphsage, self).__init__()
        self.adjlist = {}
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout
        shapes = [num_features] + hidden_size + [num_classes]
        # print(shapes)
        self.convs = nn.ModuleList(
            [
                GraphSAGELayer(shapes[layer], shapes[layer+1])
                for layer in range(num_layers)
            ]
        )

    # @profile
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            edge_index_sp = self.sampling(edge_index, self.sample_size[i]).to(x.device)
            x = self.convs[i](x, edge_index_sp)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)
