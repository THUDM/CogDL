import math
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

@register_model("fastgcn")
class FastGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--sample-size", type=int, nargs='+', default=[512,256,256])
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.sample_size,
        )

    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout, sample_size):
        super(FastGCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sample_size = sample_size

        shapes = [num_features] + [hidden_size] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList(
            [
                GraphConvolution(shapes[layer], shapes[layer + 1])
                for layer in range(num_layers)
            ]
        )

    def set_adj(self, edge_index, num_nodes):
        self.adj = collections.defaultdict(list)
        for i in range(edge_index.shape[1]):
            self.adj[edge_index[0, i]].append(edge_index[1, i])
            self.adj[edge_index[1, i]].append(edge_index[0, i])
    
    def _sample_one_layer(self, sampled, sample_size):
        total = []
        for node in sampled:
            total.extend(self.adj[node])
        total = list(set(total))
        if sample_size < len(total):
            total = random.sample(total, sample_size)
        return total

    def _generate_adj(self, sample1, sample2):
        edgelist = []
        mapping = {}
        for i in range(len(sample1)):
            mapping[sample1[i]] = i

        for i in range(len(sample2)):
            nodes = self.adj[sample2[i]]
            for node in nodes:
                if node in mapping:
                    edgelist.append([mapping[node], i])
        edgetensor = torch.LongTensor(edgelist)
        valuetensor = torch.ones(edgetensor.shape[0]).float()
        t = torch.sparse_coo_tensor(
            edgetensor.t(), valuetensor, (len(sample1), len(sample2))
        )
        return t

    def sampling(self, x, v):
        all_support = [[] for _ in range(self.num_layers)]
        sampled = v.detach().cpu().numpy()
        for i in range(self.num_layers - 1, -1, -1):
            cur_sampled = self._sample_one_layer(sampled, self.sample_size[i])
            all_support[i] = self._generate_adj(sampled, cur_sampled).to(x.device)
            sampled = cur_sampled

        return x[torch.LongTensor(sampled).to(x.device)], all_support, 0

    def forward(self, x, adj):
        for index, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, adj[index]))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj[-1])
        return F.log_softmax(x, dim=1)
