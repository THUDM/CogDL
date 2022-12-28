import numpy as np
import torch.nn as nn
from cogdl.layers import STGATConvLayer
from cogdl.layers.gcn_layerii import GCNLayerST

from .. import BaseModel
import torch
import torch.nn.functional as F



class STGAT(BaseModel):
    """
    Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
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
        parser.add_argument("--num_nodes", type=int, default=50)
        # fmt: on


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.n_his,
            args.n_pred,
        )

    def __init__(self, in_channels, out_channels, n_nodes=288, heads=8, dropout=0.0):
        """
        Initialize the ST-GAT model
        :param in_channels Number of input channels
        :param out_channels Number of output channels
        :param n_nodes Number of nodes in the graph
        :param heads Number of attention heads to use in graph
        :param dropout Dropout probability on output of Graph Attention Network
        """
        super(STGAT, self).__init__()
        self.n_his = in_channels
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 1
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # single graph attentional layer with 8 attention heads
        # self.gat = STGATConvLayer(in_channels=in_channels, out_channels=in_channels,
        #     heads=heads, dropout=0, concat=False)

        # TODU: use gat_layer to build stgat in cogdl layers, now gcn_layer is used.
        self.gat = GCNLayerST(in_channels, in_channels)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x, edge_index, edge_weight, batch_size, num_feature):
        """
        Forward pass of the ST-GAT model
        :param data Data to make a pass on
        :param device Device to operate on
        """
        # apply dropout
        if torch.cuda.is_available():  
            x = torch.cuda.FloatTensor(x)
        else:
            x = torch.FloatTensor(x)


        x = self.gat(x, edge_index, edge_weight)
        x = F.dropout(x, self.dropout, training=self.training)
        batch_size = batch_size
        n_node = 288
        x = torch.reshape(x, (batch_size, n_node, num_feature))
        x = torch.movedim(x, 2, 0)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Output contains h_t for each timestep, only the last one has all input's accounted for
        x = torch.squeeze(x[-1, :, :])
        x = self.linear(x)

        # Now reshape into final output
        x = x.view(-1,n_node)
        s = x.shape

        x = torch.reshape(x, (s[0], self.n_nodes, self.n_pred))
        x = torch.reshape(x, (s[0]*self.n_nodes, self.n_pred))

        xx = x[:,-1].view(batch_size,n_node)
        return xx


