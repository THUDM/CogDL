import torch.nn as nn

from .. import BaseModel, register_model
from cogdl.utils import spmm


class SimpleGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, order=3):
        super(SimpleGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.W = nn.Linear(in_features, out_features)

    def forward(self, graph, x):
        output = self.W(x)
        for _ in range(self.order):
            output = spmm(graph, output)
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

    def forward(self, graph):
        graph.sym_norm()

        x = self.nn(graph, graph.x)
        return x

    def predict(self, data):
        return self.forward(data)
