from cogdl.utils import add_remaining_self_loops
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
    def __init__(self, num_features, out_feats, attention, activation, normalization, nhop, subheads=2, dropout=0.5,
                 node_dropout=0.5, alpha=0.2, concat=True):
        super(SrgcnHead, self).__init__()

        self.subheads = subheads
        self.concat = concat
        self.alpha = alpha
        self.nhop = nhop

        self.attention = attention(out_feats)
        self.activation = activation
        self.normalization = normalization()

        self.adaptive_enc = nn.ModuleList()

        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()

        # multi-head
        for r in range(self.subheads):
            W = nn.Parameter(torch.zeros(size=(num_features, out_feats)))
            nn.init.xavier_normal_(W.data, gain=1.414)
            self.weight.append(W)
            self.bias.append(nn.Parameter(torch.zeros(size=(out_feats,))))
            self.adaptive_enc.append(NodeAdaptiveEncoder(out_feats, dropout))

        self.dropout = dropout
        self.node_dropout = node_dropout

    def forward(self, x, edge_index, edge_attr):
        N, dim = x.shape
        # x = self.dropout(x)

        # nl_adj_mat_ind, nl_adj_mat_val = add_self_loops(edge_index, num_nodes=N)[0], edge_attr.squeeze()
        nl_adj_mat_ind = add_remaining_self_loops(edge_index, num_nodes=N)[0]
        nl_adj_mat_val = torch.ones(nl_adj_mat_ind.shape[1]).to(x.device)

        for _ in range(self.nhop-1):
            nl_adj_mat_ind, nl_adj_mat_val = spspmm(nl_adj_mat_ind, nl_adj_mat_val, nl_adj_mat_ind, nl_adj_mat_val, N,
                                                    N, N, True)

        result = []
        for i in range(self.subheads):
            h = torch.mm(x, self.weight[i])

            adj_mat_ind, adj_mat_val = nl_adj_mat_ind, nl_adj_mat_val
            h = F.dropout(h, p=self.dropout, training=self.training)

            adj_mat_ind, adj_mat_val = self.attention(h, adj_mat_ind, adj_mat_val)
            # laplacian matrix normalization
            adj_mat_val = self.normalization(adj_mat_ind, adj_mat_val, N)

            val_h = h

            for _ in range(i + 1):
                val_h = spmm(adj_mat_ind, adj_mat_val, N, N, val_h)
                # val_h = spmm(adj_mat_ind, F.dropout(adj_mat_val, p=self.node_dropout, training=self.training), N, N, val_h)

            # val_h = val_h / norm
            val_h[val_h != val_h] = 0
            val_h = val_h + self.bias[i]
            val_h = self.adaptive_enc[i](val_h)
            val_h = self.activation(val_h)
            val_h = F.dropout(val_h, p=self.dropout, training=self.training)
            result.append(val_h)
        h_res = torch.cat(result, dim=1)
        return h_res


class SrgcnSoftmaxHead(nn.Module):
    def __init__(self, num_features, out_feats, attention, activation, nhop, normalization, dropout=0.5,
                 node_dropout=0.5, alpha=0.2):
        super(SrgcnSoftmaxHead, self).__init__()

        self.alpha = alpha
        self.activation = activation
        self.nhop = nhop
        self.normalization = normalization()
        self.attention = attention(out_feats)

        self.weight = nn.Parameter(torch.zeros(size=(num_features, out_feats)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(out_feats,)))
        self.adaptive_enc = NodeAdaptiveEncoder(out_feats, dropout)

        self.dropout = dropout
        self.node_dropout = node_dropout

    def forward(self, x, edge_index, edge_attr):
        N, dim = x.shape
        # x = self.dropout(x)

        # adj_mat_ind, adj_mat_val = add_self_loops(edge_index, num_nodes=N)[0], edge_attr.squeeze()
        adj_mat_ind = add_remaining_self_loops(edge_index, num_nodes=N)[0]
        adj_mat_val = torch.ones(adj_mat_ind.shape[1]).to(x.device)

        h = torch.mm(x, self.weight)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for _ in range(self.nhop-1):
            adj_mat_ind, adj_mat_val = spspmm(adj_mat_ind, adj_mat_val, adj_mat_ind, adj_mat_val, N, N, N, True)

        adj_mat_ind, adj_mat_val = self.attention(h, adj_mat_ind, adj_mat_val)

        # MATRIX_MUL
        # laplacian matrix normalization
        adj_mat_val = self.normalization(adj_mat_ind, adj_mat_val, N)

        val_h = h
        # N, dim = val_h.shape

        # MATRIX_MUL
        # val_h = spmm(adj_mat_ind, F.dropout(adj_mat_val, p=self.node_dropout, training=self.training), N, N, val_h)
        val_h = spmm(adj_mat_ind, adj_mat_val, N, N, val_h)

        val_h[val_h != val_h] = 0
        val_h = val_h + self.bias
        val_h = self.adaptive_enc(val_h)
        val_h = F.dropout(val_h, p=self.dropout, training=self.training)
        # val_h = self.activation(val_h)
        return val_h


@register_model("srgcn")
class SRGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument('--num-heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--node-dropout', type=float, default=0.5)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--lr', type=float, default=0.005)
        parser.add_argument('--subheads', type=int, default=1)
        parser.add_argument('--attention-type', type=str, default='node')
        parser.add_argument('--activation', type=str, default='leaky_relu')
        parser.add_argument('--nhop', type=int, default=1)
        parser.add_argument('--normalization', type=str, default='row_uniform')

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            num_features=args.num_features,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes,
            dropout=args.dropout,
            node_dropout=args.node_dropout,
            nhead=args.num_heads,
            subheads=args.subheads,
            alpha=args.alpha,
            attention=args.attention_type,
            activation=args.activation,
            nhop=args.nhop,
            normalization=args.normalization
        )

    def __init__(self, num_features, hidden_size, num_classes, attention, activation, nhop, normalization, dropout,
                 node_dropout, alpha, nhead, subheads):
        super(SRGCN, self).__init__()
        attn_f = act_attention(attention)
        activate_f = act_map(activation)
        norm_f = act_normalization(normalization)
        self.attentions = [SrgcnHead(num_features=num_features,
                                     out_feats=hidden_size,
                                     attention=attn_f,
                                     activation=activate_f,
                                     nhop=nhop,
                                     normalization=norm_f,
                                     subheads=subheads,
                                     dropout=dropout,
                                     node_dropout=node_dropout,
                                     alpha=alpha,
                                     concat=True)
                           for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SrgcnSoftmaxHead(num_features=hidden_size * nhead * subheads,
                                        out_feats=num_classes,
                                        attention=attn_f,
                                        activation=activate_f,
                                        normalization=act_normalization('row_softmax'),
                                        nhop=nhop,
                                        dropout=dropout,
                                        node_dropout=node_dropout)

    def forward(self, batch):
        x = torch.cat([att(batch.x, batch.edge_index, batch.edge_attr) for att in self.attentions], dim=1)
        x = F.elu(x)
        x = self.out_att(x, batch.edge_index, batch.edge_attr)
        return F.log_softmax(x, dim=1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data)

