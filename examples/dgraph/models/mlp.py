import torch.nn as nn
from cogdl.layers import  MLP as MLPLayer
from cogdl.data import Graph
import torch.nn.functional as F

from cogdl.models import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        dropout=0.0,
        activation="relu",
        norm=None,
        act_first=False,
        bias=True,
    ):
        super(MLP, self).__init__()
        self.nn = MLPLayer(in_feats, out_feats, hidden_size, num_layers, dropout, activation, norm, act_first, bias)

    def reset_parameters(self):
        self.nn.reset_parameters()

    def forward(self, x):
        if isinstance(x, Graph):
            x = x.x
        #return self.nn(x)
        return F.log_softmax(self.nn(x), dim=-1)

    def predict(self, data):
        return self.forward(data.x)
