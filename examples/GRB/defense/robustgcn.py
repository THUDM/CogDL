"""Torch module for RobustGCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GCNLayer
from cogdl.models import BaseModel
import cogdl.utils.grb_utils as utils
from cogdl.utils.grb_utils import getGRBGraph, updateGraph, RobustGCNAdjNorm
import copy
import types


class RobustGCN(BaseModel):
    r"""

    Description
    -----------
    Robust Graph Convolutional Networks (`RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__)

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``RobustAdjNorm``.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument("--num-features", type=int)
    #     parser.add_argument("--num-classes", type=int)
    #     parser.add_argument("--num-layers", type=int, default=2)
    #     parser.add_argument("--hidden-size", type=int, default=64)
    #     parser.add_argument("--dropout", type=float, default=0.5)
    #     parser.add_argument("--feat-norm", type=types.FunctionType, default=None)
    #     parser.add_argument("--adj-norm", type=types.FunctionType, default=RobustGCNAdjNorm)
    #     # fmt: on

    # @classmethod
    # def build_model_from_args(cls, args):
    #     return cls(
    #         args.num_features,
    #         args.hidden_size,
    #         args.num_classes,
    #         args.num_layers,
    #         args.dropout,
    #         args.feat_norm,
    #         args.adj_norm,
    #     )

    def __init__(
        self, in_feats, hidden_size, out_feats, num_layers, dropout=0.0, feat_norm=None, adj_norm_func=RobustGCNAdjNorm
    ):
        super(RobustGCN, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_size) is int:
            hidden_size = [hidden_size] * (num_layers - 1)
        elif type(hidden_size) is list or type(hidden_size) is tuple:
            assert len(hidden_size) == (num_layers - 1), "Incompatible sizes between hidden_size and n_layers."
        n_features = [in_feats] + hidden_size + [out_feats]

        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                RobustGCNConv(
                    n_features[i],
                    n_features[i + 1],
                    act0=self.act0,
                    act1=self.act1,
                    initial=True if i == 0 else False,
                    dropout=dropout if i != num_layers - 1 else 0.0,
                )
            )

    def forward(self, graph):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : list of torch.SparseTensor
            List of sparse tensor of adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """
        adj, x = getGRBGraph(graph)
        adj0, adj1 = copy.deepcopy(adj), copy.deepcopy(adj)
        mean = x
        var = x
        for layer in self.layers:
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1)
        sample = torch.randn(var.shape).to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output


class RobustGCNConv(nn.Module):
    r"""

    Description
    -----------
    RobustGCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    act0 : func of torch.nn.functional, optional
        Activation function. Default: ``F.elu``.
    act1 : func of torch.nn.functional, optional
        Activation function. Default: ``F.relu``.
    initial : bool, optional
        Whether to initialize variance.
    dropout : float, optional
            Rate of dropout. Default: ``0.0``.

    """

    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.0):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv = nn.Linear(in_features, out_features)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, mean, var=None, adj0=None, adj1=None):
        r"""

        Parameters
        ----------
        mean : torch.Tensor
            Tensor of mean of input features.
        var : torch.Tensor, optional
            Tensor of variance of input features. Default: ``None``.
        adj0 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 0. Default: ``None``.
        adj1 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 1. Default: ``None``.

        Returns
        -------

        """
        mean = self.mean_conv(mean)
        if self.initial:
            var = mean * 1
        else:
            var = self.var_conv(var)
        mean = self.act0(mean)
        var = self.act1(var)
        attention = torch.exp(-var)

        mean = mean * attention
        var = var * attention * attention
        mean = torch.spmm(adj0, mean)
        var = torch.spmm(adj1, var)
        if self.dropout:
            mean = self.act0(mean)
            var = self.act1(var)
            if self.dropout is not None:
                mean = self.dropout(mean)
                var = self.dropout(var)

        return mean, var
