import torch.nn as nn

from cogdl.utils import spmm


class SGCLayer(nn.Module):
    def __init__(self, in_features, out_features, order=3):
        super(SGCLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.W = nn.Linear(in_features, out_features)

    def forward(self, graph, x):
        output = self.W(x)
        for _ in range(self.order):
            output = spmm(graph, output)
        return output
