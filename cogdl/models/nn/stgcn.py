import numpy as np
import torch.nn as nn
from cogdl.layers import STConvLayer
from .. import BaseModel
import torch


class STGCN(BaseModel):
    """
    Args:
        in_features (int) : Number of input features.
        out_features (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        # model parameters

        parser.add_argument("--channel_size_list", default=np.array([[1, 16, 64], [64, 16, 64]]))
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--K", type=int, default=3)
        parser.add_argument("--normalization", type=str, default='sym')
        parser.add_argument("--num_nodes", type=int, default=288)
        # fmt: on


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.channel_size_list,
            args.kernel_size,
            args.num_layers,
            args.K,
            args.n_his,
            args.normalization,
            args.num_nodes,
        )


    def __init__(self,
                channel_size_list,
                kernel_size,
                num_layers,
                K,
                window_size,
                normalization,
                num_nodes,
                device = 'cuda',
                bias=True):

        super(STGCN, self).__init__()
        self.layers = nn.ModuleList([])
        # add STConv blocks
        for layer in range(num_layers):
            input_size, hidden_size, output_size = \
            channel_size_list[layer][0], channel_size_list[layer][1], \
            channel_size_list[layer][2]
            self.layers.append(STConvLayer(num_nodes, input_size, hidden_size, \
                                      output_size, kernel_size, K, \
                                      normalization, bias))

        # add output layer
        self.layers.append(OutputLayer(channel_size_list[-1][-1], \
                                       window_size - 2 * num_layers * (kernel_size - 1), \
                                       num_nodes))
        # CUDA if available
        if torch.cuda.is_available():
            for layer in self.layers:
                layer = layer.to(device)


    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers[:-1]:
          x = layer(x, edge_index, edge_weight)
        out_layer = self.layers[-1]
        x = x.permute(0, 3, 1, 2)
        x = out_layer(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = FullyConnLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        x_out = self.fc(x_t2)
        return x_out


class FullyConnLayer(nn.Module):
    def __init__(self, c):
        super(FullyConnLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)


