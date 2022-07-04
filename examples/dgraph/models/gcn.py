import torch.nn as nn
from cogdl.layers import GCNLayer
import torch.nn.functional as F

from cogdl.models import BaseModel


class GCN(BaseModel):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
        in_features (int) : Number of input features.
        out_features (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout=0.0,
        activation="relu",
        residual=False,
        norm=None,
    ):
        super(GCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                    residual=residual if i != num_layers - 1 else None,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def embed(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers - 1):
            h = self.layers[i](graph, h)
        return h

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        return F.log_softmax(h, dim=-1)
