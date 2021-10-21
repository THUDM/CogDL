from cogdl.layers import SGCLayer

from .. import BaseModel


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
        self.nn = SGCLayer(in_feats, out_feats)
        self.cache = dict()

    def forward(self, graph):
        graph.sym_norm()

        x = self.nn(graph, graph.x)
        return x

    def predict(self, data):
        return self.forward(data)
