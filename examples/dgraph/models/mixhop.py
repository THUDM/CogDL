import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import MixHopLayer
from cogdl.models import BaseModel


class MixHop(BaseModel):
    def __init__(self, num_features, num_classes, dropout, layer1_pows, layer2_pows):
        super(MixHop, self).__init__()

        self.dropout = dropout

        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        layer_pows = [layer1_pows, layer2_pows]

        shapes = [num_features] + [sum(layer1_pows), sum(layer2_pows)]

        self.mixhops = nn.ModuleList(
            [MixHopLayer(shapes[layer], [0, 1, 2], layer_pows[layer]) for layer in range(len(layer_pows))]
        )
        self.fc = nn.Linear(shapes[-1], num_classes)

    def forward(self, graph):
        x = graph.x
        for mixhop in self.mixhops:
            x = F.relu(mixhop(graph, x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x

    def predict(self, data):
        return self.forward(data)
