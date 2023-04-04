import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.data import Graph
from cogdl.layers import SAGELayer

from .. import BaseModel


def sage_sampler(adjlist, edge_index, num_sample):
    if adjlist == {}:
        row, col = edge_index
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        for i in zip(row, col):
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

    edge_idx = torch.as_tensor(sample_list, dtype=torch.long).t()
    return edge_idx


class Graphsage(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, nargs='+', default=[128])
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[10, 10])
        parser.add_argument("--dropout", type=float, default=0.5)
        # parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--aggr", type=str, default="mean")
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
            args.aggr,
        )

    def sampling(self, edge_index, num_sample):
        return sage_sampler(self.adjlist, edge_index, num_sample)

    def __init__(self, num_features, num_classes, hidden_size, num_layers, sample_size, dropout, aggr):
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
        self.convs = nn.ModuleList(
            [SAGELayer(shapes[layer], shapes[layer + 1], aggr=aggr) for layer in range(num_layers)]
        )

    def mini_forward(self, graph):
        x = graph.x
        for i in range(self.num_layers):
            edge_index_sp = self.sampling(graph.edge_index, self.sample_size[i]).to(x.device)
            with graph.local_graph():
                graph.edge_index = edge_index_sp
                x = self.convs[i](graph, x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        self.adjlist.clear()
        return x

    def forward(self, *args):
        if isinstance(args[0], Graph):
            return self.mini_forward(*args)
        else:
            device = next(self.parameters()).device
            x, adjs = args
            for i, (src_id, graph, size) in enumerate(adjs):
                graph = graph.to(device)
                output = self.convs[i](graph, x)
                x = output[: size[1]]
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

    def inference(self, x_all, data_loader):
        device = next(self.parameters()).device
        for i in range(len(self.convs)):
            output = []
            for src_id, graph, size in data_loader:
                x = x_all[src_id].to(device)
                graph = graph.to(device)
                x = self.convs[i](graph, x)
                x = x[: size[1]]
                if i != self.num_layers - 1:
                    x = F.relu(x)
                output.append(x.cpu())
            x_all = torch.cat(output, dim=0)
        return x_all


class SAGE(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--aggr", type=str, default="mean")
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--normalize", action="store_true")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.aggr,
            args.dropout,
            args.norm,
            args.activation,
            args.normalize if hasattr(args, "normalize") else False,
            args.actnn,
        )

    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        aggr="mean",
        dropout=0.5,
        norm=None,
        activation=None,
        normalize=False,
        actnn=False,
    ):
        super(SAGE, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.num_layers = num_layers
        Layer = SAGELayer
        if actnn:
            try:
                from cogdl.layers.actsage_layer import ActSAGELayer
            except Exception:
                print("Please install the actnn library first.")
                exit(1)
            Layer = ActSAGELayer
        self.layers = nn.ModuleList(
            [
                Layer(
                    shapes[i],
                    shapes[i + 1],
                    aggr=aggr,
                    normalize=normalize if i != num_layers - 1 else False,
                    dropout=dropout,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, graph):
        x = graph.x
        for layer in self.layers:
            x = layer(graph, x)
        return x

class UnsupGraphsage(Graphsage):
    def __init__(self, num_features, num_classes, hidden_size, num_layers, sample_size, dropout, aggr):
        super(Graphsage, self).__init__()
        assert num_layers == len(sample_size)
        self.adjlist = {}
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout
        shapes = [num_features] + hidden_size * num_layers
        self.convs = nn.ModuleList(
            [SAGELayer(shapes[layer], shapes[layer + 1], aggr=aggr) for layer in range(num_layers)]
        )