import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.layers.gcn_layerii import GCNLayerST

# from torch_geometric.nn import GCNConv,ChebConv



"""
一层时间卷积
一层图谱卷积
一层时间卷积
返回 batch * n_hid - (keral_size-1)*2 * num_nodes * out_channels
"""


class STConvLayer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConvLayer, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )


        self._graph_conv = GCNLayerST(
            hidden_channels,
            hidden_channels,
        )

        # self._graph_conv = ChebConv(
        #     in_channels=hidden_channels,
        #     out_channels=hidden_channels,
        #     K=K,
        #     normalization=normalization,
        #     bias=bias,
        # )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        # 归一化层
        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(
        self,
        X: torch.FloatTensor, # 30*20*50*1
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """

        # X = 30*20*50*1
        T_0 = self._temporal_conv1(X)
        # 30*18*50*16

        #  torch.zeros_like:生成和括号内变量维度维度一致的全是零的内容

        # 不能覆盖，因为会丢失梯度
        T = torch.zeros_like(T_0).to(T_0.device)
        for b in range(T_0.size(0)):
            # 对每一个样本
            for t in range(T_0.size(1)):
                # 对 每一个通道 的 特征矩阵 用于 类似于 GCN 的操作， 需要输入特征矩阵，领接举证以及边的权重，
                T[b][t] = self._graph_conv(T_0[b][t] ,edge_index, edge_weight)


        # 30*18*50*16
        T = F.relu(T)

        # 30*18*50*16
        T = self._temporal_conv2(T)
        #  30*16*50*64

        # 30*16*50*64 --> 30*50*16*64
        # 对每一行
        T = T.permute(0, 2, 1, 3)

        # batch归一化
        T = self._batch_norm(T)

        # 30*16*50*64 <-- 30*50*16*64
        T = T.permute(0, 2, 1, 3)
        # 返回 30*16*50*64
        return T


# 时间卷积 包含 3 层卷积
class TemporalConv(nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """
    # [input_size, output_size] = [1, 16] ,  K = 3
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        # X = 30*20*50*1 --> 30*1*50*20    转化为 一个通道 样本大小为 50*20 的 特征矩阵。
        X = X.permute(0, 3, 2, 1)
        #  输入 30*1*50*20
        P = self.conv_1(X)
        #  输出 30*16*50*18
        PP = self.conv_2(X)
        #  输出 30*16*50*18
        Q = torch.sigmoid(PP)
        #  输出 30*16*50*18
        PQ = P * Q # 门控机制
        #  输出 30*16*50*18
        PPP = self.conv_3(X)
        #  输出 30*16*50*18

        H = F.relu(PQ + PPP) # 记忆加原来的信息 并激活
        #  输出 30*16*50*18
        H = H.permute(0, 3, 2, 1)

        # 30*16*50*18 --> 30*18*50*16 # 20 维衰减为 18 维，通道变为 16
        return H
