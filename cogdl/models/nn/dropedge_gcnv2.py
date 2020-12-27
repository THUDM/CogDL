import numpy as np
import torch
import torch.nn.functional as F
from .. import BaseModel, register_model
from .gcn import GraphConvolution
from cogdl.utils import add_remaining_self_loops, spmm, add_self_loops


def drop_edge(adj, adj_values, rate):
    num_edge = adj.shape[1]
    index_edge = np.arange(num_edge)
    np.random.shuffle(index_edge)
    select_edge = np.sort(index_edge[:int((1 - rate) * num_edge)])
    new_adj = adj[:, select_edge]
    new_adj_values = adj_values[select_edge]
    return new_adj, new_adj_values


def bingge_norm_adj(adj, adj_values, num_nodes):
    adj, adj_values = add_self_loops(adj, adj_values, 1, num_nodes)
    deg = spmm(adj, adj_values, torch.ones(num_nodes, 1).to(adj.device)).squeeze()
    deg_sqrt = deg.pow(-1 / 2)
    adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
    row, col = adj[0], adj[1]
    mask = row != col
    adj_values[row[mask]] += 1
    return adj, adj_values


def aug_norm_adj(adj, adj_values, num_nodes):
    adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, num_nodes)
    deg = spmm(adj, adj_values, torch.ones(num_nodes, 1).to(adj.device)).squeeze()
    deg_sqrt = deg.pow(-1 / 2)
    adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
    return adj, adj_values


def get_normalizer(normalization):
    normalizer_dict = dict(AugNorm=aug_norm_adj,
                           BinggeNorm=bingge_norm_adj)
    if not normalization in normalizer_dict:
        raise NotImplementedError
    return normalizer_dict[normalization]


@register_model("dropedge_gcn")
class dropedge_gcn(BaseModel):
    r"""The DropEdge GCN model from the `"DROPEDGE: TOWARDS DEEP GRAPH CONVOLUTIONAL NETWORKS ON NODE CLASSIFICATION"
    <https://arxiv.org/abs/1907.10903>`_ paper

    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        #DropEdge
        parser.add_argument("--dropedge", type=float, default=0.0)
        parser.add_argument("--normalization", type=str, default="AugNorm")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout,args.dropedge,
                   args.normalization)

    def __init__(self, nfeat, nhid, nclass, dropout,dropedge,normalization):
        super(dropedge_gcn, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.dropedge = dropedge
        self.normalization = normalization
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):
        device = x.device
        adj_values = torch.ones(adj.shape[1]).to(device)
        adj, adj_values = drop_edge(adj,adj_values,self.dropedge)
        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        adj, adj_values = get_normalizer(self.normalization)(adj, adj_values, x.shape[0])
        deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        deg_sqrt = deg.pow(-1 / 2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj, adj_values))
        # h1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, adj_values)

        # x = F.relu(x)
        # x = torch.sigmoid(x)
        # return x
        # h2 = x
        return F.log_softmax(x, dim=-1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
