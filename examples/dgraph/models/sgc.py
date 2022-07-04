from cogdl.layers import SGCLayer
import torch.nn.functional as F

from cogdl.models import BaseModel


class sgc(BaseModel):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(sgc, self).__init__()
        self.nn1 = SGCLayer(in_feats, hidden_size)
        self.nn2 = SGCLayer(hidden_size, out_feats)

        self.cache = dict()

    def reset_parameters(self):
        self.nn1.W.reset_parameters()
        self.nn2.W.reset_parameters()

    def forward(self, graph):
        graph.sym_norm()

        x = self.nn1(graph, graph.x)
        x = self.nn2(graph, x)

        return F.log_softmax(x, dim=-1)

    def predict(self, data):
        return self.forward(data)

