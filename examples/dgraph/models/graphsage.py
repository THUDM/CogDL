import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.data import Graph
from cogdl.layers import SAGELayer

from cogdl.models import BaseModel


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

    def reset_parameters(self):
        for layer in self.layers:
            layer.fc.reset_parameters()

    def forward(self, graph):
        x = graph.x
        for layer in self.layers:
            x = layer(graph, x)
        return F.log_softmax(x, dim=-1)

