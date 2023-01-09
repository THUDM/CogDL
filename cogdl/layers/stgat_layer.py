import torch
import torch.nn as nn
from cogdl.layers.gat_layerii import GATLayerST



class STGATConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        dropout=0,
        concat=False
    ):
        super(STGATConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self._graph_conv = GATLayerST(in_channels, out_channels, nhead=1, alpha=0.2, attn_drop=0.5, activation=None, residual=False, norm=None)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        x = self._graph_conv(X ,edge_index, edge_weight)

        return x


