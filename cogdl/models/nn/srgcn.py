import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm, spspmm
from torch_geometric.utils import add_self_loops

from cogdl.layers.srgcn_module import *
from .. import BaseModel, register_model


class NodeAdaptiveEncoder(nn.Module):
    def __init__(self, num_features, dropout=0.5):
        super(NodeAdaptiveEncoder, self).__init__()
        # self.fc = nn.Linear(num_features, 1, bias=True)
        self.fc = nn.Parameter(torch.zeros(size=(num_features, 1)))
        nn.init.xavier_normal_(self.fc.data, gain=1.414)
        self.bf = nn.Parameter(torch.zeros(size=(1,)))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # h = self.fc(x)
        # h = F.sigmoid(h)
        # h = self.dropout(h)
        h = torch.mm(x, self.fc) + self.bf
        h = F.sigmoid(h)
        h = self.dropout(h)

        return torch.where(x < 0, torch.zeros_like(x), x) + h * torch.where(x > 0, torch.zeros_like(x), x)


class SrgcnHead(nn.Module):
    def __init__(self, num_features, out_feats, attention, normalization, nhop, dropout=0.5,
                 node_dropout=0.5, alpha=0.2, concat=True):
        super(SrgcnHead, self).__init__()

        self.concat = concat
        self.alpha = alpha
        self.nhop = nhop

        # self.attention = attention(out_feats)
        self.normalization = normalization()
        self.attentions = nn.ModuleList()
        self.adj_normalization = act_normalization("row_uniform")()

        self.adaptive_enc = nn.ModuleList()
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()

        # multi-hop
        for r in range(self.nhop):
            W = nn.Parameter(torch.zeros(size=(num_features, out_feats)))
            nn.init.xavier_normal_(W.data, gain=1.414)
            self.weight.append(W)
            self.bias.append(nn.Parameter(torch.zeros(size=(out_feats,))))
            self.adaptive_enc.append(NodeAdaptiveEncoder(out_feats, dropout))
            self.attentions.append(attention(out_feats))

        self.dropout = dropout
        self.node_dropout = node_dropout

    def forward(self, x, edge_index, edge_attr):
        N, dim = x.shape
        x = F.dropout(x, p=self.dropout, training=self.training)  # droprate = 0.6

        # nl_adj_mat_ind, nl_adj_mat_val = add_self_loops(edge_index, num_nodes=N)[0], edge_attr.squeeze()

        result = []
        for i in range(self.nhop):
            h = torch.mm(x, self.weight[i])
            h = F.dropout(h, p=self.dropout, training=self.training)
            # laplacian matrix normalization
            adj_mat_ind, adj_mat_val = edge_index, edge_attr

            # edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

            for _ in range(i):
                # scatter_add
                # h_selected = torch.index_select(val_h, 0, edge_index[1])
                # val_h = torch.zeros_like(h).scatter_add_(dim=0, index=edge_index[0].unsqueeze(1).repeat(1, dim), src=h_selected)
                adj_mat_ind, adj_mat_val = spspmm(adj_mat_ind,
                                                  adj_mat_val,
                                                  # F.dropout(adj_mat_val, p=self.dropout, training=self.training),
                                                  edge_index, edge_attr, N, N, N, True)

            adj_mat_ind, adj_mat_val = self.attentions[i](h, adj_mat_ind, adj_mat_val)
            adj_mat_val = self.normalization(adj_mat_ind, adj_mat_val, N)
            val_h = spmm(adj_mat_ind,
                         # adj_mat_val,
                         F.dropout(adj_mat_val, p=self.dropout,training=self.training),
                         N, N, h)

            # val_h = val_h / norm
            val_h[val_h != val_h] = 0
            val_h = val_h + self.bias[i]
            val_h = self.adaptive_enc[i](val_h)
            result.append(val_h)
        h_res = torch.cat(result, dim=1)
        return h_res



class SrgcnSoftmaxHead(nn.Module):
    def __init__(self, num_features, out_feats, attention, nhop, normalization, dropout=0.5,
                 node_dropout=0.5, alpha=0.2):
        super(SrgcnSoftmaxHead, self).__init__()

        self.alpha = alpha
        self.nhop = nhop
        self.normalization = normalization()
        self.attention = attention(out_feats)
        self.adj_normalization = act_normalization("row_uniform")()

        self.weight = nn.Parameter(torch.zeros(size=(num_features, out_feats)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(out_feats,)))
        self.adaptive_enc = NodeAdaptiveEncoder(out_feats, dropout)

        self.dropout = dropout
        self.node_dropout = node_dropout

    def forward(self, x, edge_index, edge_attr):
        N, dim = x.shape
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = torch.mm(x, self.weight)
        h = F.dropout(h, p=self.dropout, training=self.training)

        adj_mat_ind, adj_mat_val = self.attention(h, edge_index, edge_attr)

        # laplacian matrix normalization
        adj_mat_val = self.normalization(adj_mat_ind, adj_mat_val, N)

        val_h = spmm(adj_mat_ind, F.dropout(adj_mat_val, p=self.dropout, training=self.training), N, N, h)
        # val_h = spmm(adj_mat_ind, adj_mat_val, N, N, val_h)

        val_h[val_h != val_h] = 0
        val_h = val_h + self.bias
        val_h = self.adaptive_enc(val_h)
        return val_h



@register_model("srgnn")
class SRGCN(BaseModel):
    """
        Single GNN layer, previously designed for multi-layer SRGCN
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument('--num-heads', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--node-dropout', type=float, default=0.5)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--lr', type=float, default=0.005)
        parser.add_argument('--num-hops', type=int, default=1)
        parser.add_argument('--normalization', type=str, default='row_uniform')
        parser.add_argument('--activation', type=str, default='leaky_relu')
        parser.add_argument('--attention-type', type=str, default='identity')
        parser.add_argument('--num-heads', type=int, default=4)
        parser.add_argument('--num-hops', type=int, default=1)
        parser.add_argument('--adj-normalization', type=str, default='row_uniform')

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            num_features=args.num_features,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            node_dropout=args.node_dropout,
            nhead=args.num_heads,
            alpha=args.alpha,
            attention=args.attention_type,
            activation=args.activation,
            nhop=args.num_hops,
            normalization=args.normalization,
            adj_normalization=args.adj_normalization,
        )

    def __init__(self, num_features, hidden_size, attention, activation, nhop, normalization, dropout,
                 node_dropout, alpha, nhead, adj_normalization):
        super(SRGCN, self).__init__()
        attn_f = act_attention(attention)
        norm_f = act_normalization(normalization)
        self.activation = act_map(activation)
        self.adj_normalization = act_normalization(adj_normalization)()
        self.attentions = [SrgcnHead(num_features=num_features,
                                     out_feats=hidden_size,
                                     attention=attn_f,
                                     nhop=nhop,
                                     normalization=norm_f,
                                     dropout=dropout,
                                     node_dropout=node_dropout,
                                     alpha=alpha,
                                     concat=True)
                           for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # self.out_att = SrgcnSoftmaxHead(num_features=hidden_size * nhead * nhop,
        #                                 out_feats=num_classes,
        #                                 attention=attn_f,
        #                                 normalization=act_normalization('row_softmax'),
        #                                 nhop=nhop,
        #                                 dropout=dropout,
        #                                 node_dropout=node_dropout)

    def forward(self, x, edge_index, edge_attr=None):
        N = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes=N)[0]
        edge_attr = self.adj_normalization(edge_index, torch.ones(edge_index.shape[1]).to(x.device), N)

        x = torch.cat([att(x, edge_index, edge_attr) for att in self.attentions], dim=1)
        x = self.activation(x)
        # x = self.out_att(x, edge_index, edge_attr)
        # x = self.activation(x)
        # return F.log_softmax(x, dim=1)
        return x


@register_model("srgcn")
class SSRGCN(BaseModel):
    """
        original SRGCN: 2-layer multi-hop SRGCN
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument('--num-heads', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--node-dropout', type=float, default=0.5)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--lr', type=float, default=0.005)
        parser.add_argument('--num-hops', type=int, default=1)
        parser.add_argument('--normalization', type=str, default='row_uniform')
        parser.add_argument('--activation', type=str, default='leaky_relu')
        parser.add_argument('--attention-type', type=str, default='identity')
        parser.add_argument('--num-heads', type=int, default=4)
        parser.add_argument('--num-hops', type=int, default=1)
        parser.add_argument('--adj-normalization', type=str, default='row_uniform')


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            num_features=args.num_features,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            node_dropout=args.node_dropout,
            nhead=args.num_heads,
            alpha=args.alpha,
            attention=args.attention_type,
            attention_att=args.attention_type_att,
            activation=args.activation,
            nhop=args.num_hops,
            normalization=args.normalization,
            adj_normalization=args.adj_normalization,
        )

    def __init__(self, num_features, hidden_size, num_classes, attention, attention_att, activation, nhop, normalization, dropout,
                 node_dropout, alpha, nhead, adj_normalization):
        super(SSRGCN, self).__init__()
        attn_f = act_attention(attention)
        attn_att_f = act_attention(attention_att)
        norm_f = act_normalization(normalization)
        self.activation = act_map(activation)
        self.adj_normalization = act_normalization(adj_normalization)()
        self.attentions = [SrgcnHead(num_features=num_features,
                                     out_feats=hidden_size,
                                     attention=attn_f,
                                     nhop=nhop,
                                     normalization=norm_f,
                                     dropout=dropout,
                                     node_dropout=node_dropout,
                                     alpha=alpha,
                                     concat=True)
                           for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SrgcnSoftmaxHead(num_features=hidden_size * nhead * nhop,
                                        out_feats=num_classes,
                                        attention=attn_att_f,
                                        normalization=act_normalization('row_softmax'),
                                        nhop=nhop,
                                        dropout=dropout,
                                        node_dropout=node_dropout)

    def forward(self, x, edge_index, edge_attr=None):
        N = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes=N)[0]
        edge_attr = self.adj_normalization(edge_index, torch.ones(edge_index.shape[1]).to(x.device), N)

        x = torch.cat([att(x, edge_index, edge_attr) for att in self.attentions], dim=1)
        x = self.activation(x)
        x = self.out_att(x, edge_index, edge_attr)
        x = self.activation(x)
        return F.log_softmax(x, dim=1)
