import torch
import torch.nn as nn

from .. import BaseModel, register_model
from cogdl.utils import add_remaining_self_loops, symmetric_normalization, spmm


class SimpleGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, order=3):
        super(SimpleGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.W = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index, edge_attr=None):
        output = self.W(x)
        for _ in range(self.order):
            output = spmm(edge_index, edge_attr, output)
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
        self.nn = SimpleGraphConvolution(in_feats, out_feats)
        self.cache = dict()

    def forward(self, x, edge_index):
        flag = str(edge_index.shape[1])
        if flag not in self.cache:
            edge_attr = torch.ones(edge_index.shape[1]).to(x.device)
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, 1, x.shape[0])
            edge_attr = symmetric_normalization(x.shape[0], edge_index, edge_attr)
            self.cache[flag] = (edge_index, edge_attr)
        edge_index, edge_attr = self.cache[flag]

        x = self.nn(x, edge_index, edge_attr)
        return x

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
