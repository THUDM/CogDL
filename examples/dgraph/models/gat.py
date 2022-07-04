import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GATLayer
from cogdl.models import BaseModel


class GAT(BaseModel):
    r"""The GAT model from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
        alpha (float) : Coefficient of leaky_relu.
        nheads (int) : Number of attention heads.
    """
    def __init__(
        self,
        in_feats,
        hidden_size,
        out_features,
        num_layers,
        dropout=0.5,
        attn_drop=0.5,
        alpha=0.2,
        nhead=2,
        residual=False,
        last_nhead=2,
        norm=None,
    ):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.attentions.append(
            GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm)
        )
        for i in range(num_layers - 2):
            self.attentions.append(
                GATLayer(
                    hidden_size * nhead,
                    hidden_size,
                    nhead=nhead,
                    attn_drop=attn_drop,
                    alpha=alpha,
                    residual=residual,
                    norm=norm,
                )
            )
        self.attentions.append(
            GATLayer(
                hidden_size * nhead,
                out_features,
                attn_drop=attn_drop,
                alpha=alpha,
                nhead=last_nhead,
                residual=False,
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def reset_parameters(self):
        for layer in self.attentions:
            layer.reset_parameters()

    def forward(self, graph):
        x = graph.x
        for i, layer in enumerate(self.attentions):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = F.log_softmax(x)
        return x

    def predict(self, graph):
        return self.forward(graph)
