"""
Modified from https://github.com/GraphSAINT/GraphSAINT
"""

import torch
from torch import nn

from cogdl.utils import spmm


F_ACT = {"relu": nn.ReLU(), "I": lambda x: x}


class SAINTLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", order=1, aggr="mean", bias="norm-nn", **kwargs):
        """
        Layer implemented here combines the GraphSAGE-mean [1] layer with MixHop [2] layer.
        We define the concept of `order`: an order-k layer aggregates neighbor information
        from 0-hop all the way to k-hop. The operation is approximately:
            X W_0 [+] A X W_1 [+] ... [+] A^k X W_k
        where [+] is some aggregation operation such as addition or concatenation.

        Special cases:
            Order = 0  -->  standard MLP layer
            Order = 1  -->  standard GraphSAGE layer

        [1]: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
        [2]: https://arxiv.org/abs/1905.00067

        Inputs:
            dim_in      int, feature dimension for input nodes
            dim_out     int, feature dimension for output nodes
            dropout     float, dropout on weight matrices W_0 to W_k
            act         str, activation function. See F_ACT at the top of this file
            order       int, see definition above
            aggr        str, if 'mean' then [+] operation adds features of various hops
                            if 'concat' then [+] concatenates features of various hops
            bias        str, if 'bias' then apply a bias vector to features of each hop
                            if 'norm' then perform batch-normalization on output features

        Outputs:
            None
        """
        super(SAINTLayer, self).__init__()
        assert bias in ["bias", "norm", "norm-nn"]
        self.order, self.aggr = order, aggr
        self.act, self.bias = F_ACT[act], bias
        self.dropout = dropout
        self.f_lin, self.f_bias = [], []
        self.offset, self.scale = [], []
        self.num_param = 0
        for o in range(self.order + 1):
            self.f_lin.append(nn.Linear(dim_in, dim_out, bias=False))
            nn.init.xavier_uniform_(self.f_lin[-1].weight)
            self.f_bias.append(nn.Parameter(torch.zeros(dim_out)))
            self.num_param += dim_in * dim_out
            self.num_param += dim_out
            self.offset.append(nn.Parameter(torch.zeros(dim_out)))
            self.scale.append(nn.Parameter(torch.ones(dim_out)))
            if self.bias == "norm" or self.bias == "norm-nn":
                self.num_param += 2 * dim_out
        self.f_lin = nn.ModuleList(self.f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params = nn.ParameterList(self.f_bias + self.offset + self.scale)
        self.f_bias = self.params[: self.order + 1]
        if self.bias == "norm":
            self.offset = self.params[self.order + 1 : 2 * self.order + 2]
            self.scale = self.params[2 * self.order + 2 :]
        elif self.bias == "norm-nn":
            final_dim_out = dim_out * ((aggr == "concat") * (order + 1) + (aggr == "mean"))
            self.f_norm = nn.BatchNorm1d(final_dim_out, eps=1e-9, track_running_stats=True)
        self.num_param = int(self.num_param)

    def _f_feat_trans(self, _feat, _id):
        feat = self.act(self.f_lin[_id](_feat) + self.f_bias[_id])
        if self.bias == "norm":
            mean = feat.mean(dim=1).view(feat.shape[0], 1)
            var = feat.var(dim=1, unbiased=False).view(feat.shape[0], 1) + 1e-9
            feat_out = (feat - mean) * self.scale[_id] * torch.rsqrt(var) + self.offset[_id]
        else:
            feat_out = feat
        return feat_out

    def forward(self, graph, x):
        """
        Inputs:
            graph           normalized adj matrix of the subgraph
            x               2D matrix of input node features

        Outputs:
            feat_out        2D matrix of output node features
        """

        feat_in = self.f_dropout(x)
        feat_hop = [feat_in]
        # generate A^i X
        for o in range(self.order):
            feat_hop.append(spmm(graph, x))
        feat_partial = [self._f_feat_trans(ft, idf) for idf, ft in enumerate(feat_hop)]
        if self.aggr == "mean":
            feat_out = feat_partial[0]
            for o in range(len(feat_partial) - 1):
                feat_out += feat_partial[o + 1]
        elif self.aggr == "concat":
            feat_out = torch.cat(feat_partial, 1)
        else:
            raise NotImplementedError
        if self.bias == "norm-nn":
            feat_out = self.f_norm(feat_out)
        return feat_out
