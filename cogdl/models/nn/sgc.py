import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, spmm, spmm_adj


class SimpleGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.W = nn.Linear(in_features, out_features)

    def forward(self, input, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1]).float().to(input.device)
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_attr,
            (input.shape[0], input.shape[0]),
        ).to(input.device)
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output


@register_model("sgc")
class sgc(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(in_feats=args.num_features, out_feats=args.num_classes)

    def __init__(self, in_feats, out_feats):
        super(sgc, self).__init__()
        self.sgc1 = SimpleGraphConvolution(in_feats, out_feats)

    def forward(self, x, edge_index):
        device = x.device
        edge_attr = torch.ones(edge_index.shape[1]).to(device)
        edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, 1, x.shape[0])
        deg = spmm(edge_index, edge_attr, torch.ones(x.shape[0], 1).to(device)).squeeze()
        deg_sqrt = deg.pow(-1 / 2)
        edge_attr = deg_sqrt[edge_index[1]] * edge_attr * deg_sqrt[edge_index[0]]

        x = self.sgc1(x, edge_index, edge_attr)
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
