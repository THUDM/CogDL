import numpy as np
import torch.nn as nn
from cogdl.layers import STConvLayer
from .. import BaseModel


class STGCN(BaseModel):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

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

        parser.add_argument("--channel_size_list", default = np.array([[1, 16, 64], [64, 16, 64]]))
        parser.add_argument("--kernel_size", type=int, default = 3)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--K", type=int, default=3)
        parser.add_argument("--normalization", type=str, default='sym')
        parser.add_argument("--num_nodes", type=int, default=288)
        # fmt: on


    @classmethod
    def build_model_from_args(cls, args):
        # 给模型初始化参数
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
        # 用多少个 ST 卷积
        for l in range(num_layers):
            input_size, hidden_size, output_size = \
            channel_size_list[l][0], channel_size_list[l][1], \
            channel_size_list[l][2]
            self.layers.append(STConvLayer(num_nodes, input_size, hidden_size, \
                                      output_size, kernel_size, K, \
                                      normalization, bias))

        # add output layer
        self.layers.append(OutputLayer(channel_size_list[-1][-1], \
                                       window_size - 2 * num_layers * (kernel_size - 1), \
                                       num_nodes))
        # CUDA if available
        for layer in self.layers:
            layer = layer.to(device)


    def forward(self, x, edge_index, edge_weight):
        # x = batch * n_hid * nums_node * 1
        # 30*20*50*1
        # self.layers 是 [多层模型层 + 一个输出层]

        # 每一个batch 的 X 经过 多层的 模型层 输出
        # x = 30*20*50*1
        a = x
        for layer in self.layers[:-1]:
          x = layer(x, edge_index, edge_weight)
        # x = 30*12*50*64


        # x = batch * n_hid - (keral_size-1)*2 * num_nodes * out_channels
        # 模型输出后经过输出层 输出 x
        out_layer = self.layers[-1]


        # 将 X = 30*12*50*64 --> 30*64*12*50
        x = x.permute(0, 3, 1, 2)
        x = out_layer(x)

        # x = 30*1*1*50
        return x


class OutputLayer(nn.Module):
    # c = 64， T = 8，n = 50
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        # 一维的 多通道时间卷积， 输出同样的通道
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))

        # nn.LayerNorm 对每个batch 进行归一化
        # （N,C,H,W） N 个样本 C通道 每个通道 H*W
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        # 多通道转为单通道
        self.fc = FullyConnLayer(c)

    def forward(self, x):
        # 时间卷积 1
        # x = 30*64*12*50
        x_t1 = self.tconv1(x)
        # x = 30*64*1*50
        # N 个样本 C通道 每个通道 H*W     将每个样本的通道 变为 H，行变为W，列变为 C
        # 先 30*64*1*50  --> 30*1*50*64, 再将 30*1*50*64 --> 30*64*1*50
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x = 30*64*1*50
        # 时间卷积 2
        x_t2 = self.tconv2(x_ln)
        # x = 30*64*1*50

        # 返回 单通道 输出 维度 保持原来的 维度
        x_out = self.fc(x_t2)
        # x_out = 30*1*1*50
        return x_out


class FullyConnLayer(nn.Module):
    def __init__(self, c):
        super(FullyConnLayer, self).__init__()
        # 卷积核大小为 1*1
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)


