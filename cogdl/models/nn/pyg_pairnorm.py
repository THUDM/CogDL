import torch as torch
import torch.nn as nn
from cogdl.utils import row_normalization, spmm
from torch_scatter import scatter_add, scatter_max

from .. import BaseModel, register_model
from .gcn import GraphConvolution


def softmax(src, index, num_nodes=None):
    """
    sparse softmax
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList(
            [GraphAttConvOneHead(in_features, out_perhead, dropout=dropout) for _ in range(heads)]
        )

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func.
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(self.in_features, self.heads, self.out_perhead)


class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        # init
        nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain("relu"))  # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain("relu"))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input, edge_index):
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        # edge_h: 2*D x E
        edge_h = torch.cat((h[edge_index[0, :], :], h[edge_index[1, :], :]), dim=1).t()
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze())  # E
        n = len(input)
        alpha = softmax(alpha, edge_index[0], n)
        output = spmm(edge_index, self.dropout(alpha), h)  # h_prime: N x out
        # output = spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output


class PairNormNorm(nn.Module):
    def __init__(self, mode="PN", scale=1):
        """
        mode:
          "None" : No normalization
          "PN"   : Original version
          "PN-SI"  : Scale-Individually version
          "PN-SCS" : Scale-and-Center-Simultaneously version

        ("SCS"-mode is not in the paper but we found it works well in practice,
          especially for GCN and GAT.)
        PairNormNorm is typically used after each graph convolution operation.
        """
        assert mode in ["None", "PN", "PN-SI", "PN-SCS"]
        super(PairNormNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == "None":
            return x

        col_mean = x.mean(dim=0)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == "PN-SCS":
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class SGC(nn.Module):
    # for SGC we use data without normalization
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, norm_mode="None", norm_scale=10, **kwargs):
        super(SGC, self).__init__()
        self.linear = torch.nn.Linear(nfeat, nclass)
        self.norm = PairNormNorm(norm_mode, norm_scale)
        self.dropout = nn.Dropout(p=dropout)
        self.nlayer = nlayer

    def forward(self, x, edge_index, edge_attr):
        x = self.norm(x)
        for _ in range(self.nlayer):
            x = spmm(edge_index, edge_attr, x)
            x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, norm_mode="None", norm_scale=1, **kwargs):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)
        self.norm = PairNormNorm(norm_mode, norm_scale)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.dropout(x)
        x = self.gc1(x, edge_index, edge_attr)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, edge_index, edge_attr)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead, norm_mode="None", norm_scale=1, **kwargs):
        super(GAT, self).__init__()
        alpha_droprate = dropout
        self.gac1 = GraphAttConv(nfeat, nhid, nhead, alpha_droprate)
        self.gac2 = GraphAttConv(nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)
        self.norm = PairNormNorm(norm_mode, norm_scale)

    def forward(self, x, adj):
        x = self.dropout(x)  # ?
        x = self.gac1(x, adj)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gac2(x, adj)
        return x


class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0, norm_mode="None", norm_scale=1, **kwargs):
        super(DeepGCN, self).__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList(
            [GraphConvolution(nfeat if i == 0 else nhid, nhid) for i in range(nlayer - 1)]
        )
        self.out_layer = GraphConvolution(nfeat if nlayer == 1 else nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)
        self.norm = PairNormNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, edge_index, edge_attr):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, edge_index, edge_attr)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x

        x = self.dropout(x)
        x = self.out_layer(x, edge_index, edge_attr)
        return x


class DeepGAT(nn.Module):
    def __init__(
        self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0, nhead=1, norm_mode="None", norm_scale=1, **kwargs
    ):
        super(DeepGAT, self).__init__()
        assert nlayer >= 1
        alpha_droprate = dropout
        self.hidden_layers = nn.ModuleList(
            [GraphAttConv(nfeat if i == 0 else nhid, nhid, nhead, alpha_droprate) for i in range(nlayer - 1)]
        )
        self.out_layer = GraphAttConv(nfeat if nlayer == 1 else nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)
        self.norm = PairNormNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, edge_index, edge_attr=None):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, edge_index)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x

        x = self.dropout(x)
        x = self.out_layer(x, edge_index)
        return x


@register_model("pairnorm")
class PairNorm(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--pn_model", type=str, default="GCN", help="{SGC, DeepGCN, DeepGAT}")
        parser.add_argument("--hidden_layers", type=int, default=64, help="Number of hidden units.")
        parser.add_argument("--nhead", type=int, default=1, help="Number of head attentions.")
        parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate.")
        parser.add_argument("--nlayer", type=int, default=2, help="Number of layers, works for Deep model.")
        parser.add_argument("--residual", type=int, default=0, help="Residual connection")
        parser.add_argument(
            "--norm_mode", type=str, default="None", help="Mode for PairNorm, {None, PN, PN-SI, PN-SCS}"
        )
        parser.add_argument("--norm_scale", type=float, default=1.0, help="Row-normalization scale")
        parser.add_argument("--no_fea_norm", action="store_false", default=True, help="not normalize feature")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.pn_model,
            args.hidden_layers,
            args.nhead,
            args.dropout,
            args.nlayer,
            args.residual,
            args.norm_mode,
            args.norm_scale,
            args.no_fea_norm,
            args.missing_rate,
            args.num_features,
            args.num_classes,
        )

    def __init__(
        self,
        pn_model,
        hidden_layers,
        nhead,
        dropout,
        nlayer,
        residual,
        norm_mode,
        norm_scale,
        no_fea_norm,
        missing_rate,
        num_features,
        num_classes,
    ):
        super(PairNorm, self).__init__()
        self.edge_attr = None

        if pn_model == "GCN":
            self.pn_model = GCN(num_features, hidden_layers, num_classes, dropout, norm_mode, norm_scale)
        elif pn_model == "SGC":
            self.pn_model = SGC(num_features, hidden_layers, num_classes, dropout, nlayer, norm_mode, norm_scale)
        elif pn_model == "DeepGCN":
            self.pn_model = DeepGCN(
                num_features, hidden_layers, num_classes, dropout, nlayer, residual, norm_mode, norm_scale
            )
        else:
            self.pn_model = DeepGAT(
                num_features, hidden_layers, num_classes, dropout, nlayer, residual, nhead, norm_mode, norm_scale
            )

    def forward(self, x, edge_index):
        if self.edge_attr is None:
            self.edge_attr = row_normalization(x.shape[0], edge_index)
        edge_attr = self.edge_attr
        return self.pn_model(x, edge_index, edge_attr)

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
