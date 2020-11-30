import torch as torch
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn
from torch_sparse import spmm
import numpy as np
from torch_scatter import scatter_max, scatter_add
from .. import BaseModel, register_model


def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx


def softmax(src, index, num_nodes=None):
    """
        sparse softmax
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0,
                             dim_size=num_nodes)[index] + 1e-16)
    return out


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        h = torch.mm(input, self.weight)
        output = torch.spmm(adj, h)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->{})".format(
            self.in_features, self.out_features)


class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
            in_features, out_perhead, dropout=dropout) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func.
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
            self.in_features, self.heads, self.out_perhead)


class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()
        self.weight = nn.Parameter(torch.zeros(
            size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        # init
        nn.init.xavier_normal_(
            self.weight.data, gain=nn.init.calculate_gain('relu'))  # look at here
        nn.init.xavier_normal_(
            self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        # edge_h: 2*D x E
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze())  # E
        n = len(input)
        alpha = softmax(alpha, edge[0], n)
        output = spmm(edge, self.dropout(alpha), n, n, h)  # h_prime: N x out
        # output = spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output


class PairNormNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)
            PairNormNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNormNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (
                1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (
                1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class SGC(nn.Module):
    # for SGC we use data without normalization
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, norm_mode='None', norm_scale=10, **kwargs):
        super(SGC, self).__init__()
        self.linear = torch.nn.Linear(nfeat, nclass)
        self.norm = PairNormNorm(norm_mode, norm_scale)
        self.dropout = nn.Dropout(p=dropout)
        self.nlayer = nlayer

    def forward(self, x, adj):
        x = self.norm(x)
        for _ in range(self.nlayer):
            x = adj.mm(x)
            x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)
        self.norm = PairNormNorm(norm_mode, norm_scale)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = self.gc1(x, adj)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead,
                 norm_mode='None', norm_scale=1, **kwargs):
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
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGCN, self).__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i == 0 else nhid, nhid)
            for i in range(nlayer-1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer == 1 else nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.norm = PairNormNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x


class DeepGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0, nhead=1,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGAT, self).__init__()
        assert nlayer >= 1
        alpha_droprate = dropout
        self.hidden_layers = nn.ModuleList([
            GraphAttConv(nfeat if i == 0 else nhid,
                         nhid, nhead, alpha_droprate)
            for i in range(nlayer-1)
        ])
        self.out_layer = GraphAttConv(
            nfeat if nlayer == 1 else nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)
        self.norm = PairNormNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x






@register_model("pairnorm")
class PairNorm(BaseModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--pn_model', type=str,
                            default='GCN', help='{SGC, DeepGCN, DeepGAT}')
        parser.add_argument('--hidden_layers', type=int,
                            default=64, help='Number of hidden units.')
        parser.add_argument('--nhead', type=int, default=1,
                            help='Number of head attentions.')
        parser.add_argument('--dropout', type=float,
                            default=0.6, help='Dropout rate.')
        parser.add_argument('--nlayer', type=int, default=2,
                            help='Number of layers, works for Deep model.')
        parser.add_argument('--residual', type=int,
                            default=0, help='Residual connection')
        parser.add_argument('--norm_mode', type=str, default='None',
                            help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS}')
        parser.add_argument('--norm_scale', type=float,
                            default=1.0, help='Row-normalization scale')
        parser.add_argument('--no_fea_norm', action='store_false',
                            default=True, help='not normalize feature')
        parser.add_argument('--missing_rate', type=int,
                            default=0, help='missing rate, from 0 to 100')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--lr', type=float, default=0.005,
                            help='Initial learning rate.')

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.pn_model, args.hidden_layers, args.nhead, args.dropout, args.nlayer, args.residual, args.norm_mode, args.norm_scale, args.no_fea_norm, args.missing_rate, args.num_features, args.num_classes)

    def __init__(self, pn_model, hidden_layers, nhead, dropout, nlayer, residual, norm_mode, norm_scale, no_fea_norm, missing_rate, num_features, num_classes):
        super(PairNorm, self).__init__()

        print((pn_model, hidden_layers, nhead, dropout, nlayer, residual, norm_mode, norm_scale, no_fea_norm, missing_rate, num_features, num_classes))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.CrossEntropyLoss()

        self.adj = None

        if pn_model == 'GCN':
            self.pn_model = GCN(num_features, hidden_layers, num_classes,
                                dropout, norm_mode, norm_scale).to(self.device)
        elif pn_model == 'SGC':
            self.pn_model = SGC(num_features, hidden_layers, num_classes,
                                dropout, nlayer, norm_mode, norm_scale).to(self.device)
        elif pn_model == 'DeepGCN':
            self.pn_model = DeepGCN(num_features, hidden_layers, num_classes,
                                    dropout, nlayer, residual, norm_mode, norm_scale).to(self.device)
        else:
            self.pn_model = DeepGAT(num_features, hidden_layers, num_classes, dropout,
                                    nlayer, residual, nhead, norm_mode, norm_scale).to(self.device)

    def forward(self,x, edge_index):
        self.x = torch.Tensor(x).to(self.device)

        if self.adj == None:       
            n = len(self.x)
            adj = sp.csr_matrix(
                (np.ones(edge_index.shape[1]), edge_index), shape=(n, n))
            adj = adj + adj.T.multiply(adj.T > adj) - \
                adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
            adj = normalize_adj_row(adj)
            self.adj = to_torch_sparse(adj).to(self.device)

        return F.log_softmax(self.pn_model(self.x, self.adj),dim=1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )
    
    def predict(self, data):
        return self.forward(data.x, data.edge_index)

