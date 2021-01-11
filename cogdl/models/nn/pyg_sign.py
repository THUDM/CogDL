import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, to_undirected
from torch_sparse import SparseTensor

from .. import BaseModel, register_model


def get_adj(row, col, N, asymm_norm=False, set_diag=True, remove_diag=False):
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    if set_diag:
        adj = adj.set_diag()
    elif remove_diag:
        adj = adj.remove_diag()

    if not asymm_norm:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    else:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float("inf")] = 0
        adj = deg_inv.view(-1, 1) * adj

    return adj


@register_model("sign")
class MLP(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--num-features', type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument('--hidden-size', type=int, default=512)
        parser.add_argument('--num-layers', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.3)
        parser.add_argument('--dropedge-rate', type=float, default=0.2)

        parser.add_argument('--directed', action='store_true')
        parser.add_argument('--num-propagations', type=int, default=1)
        parser.add_argument('--asymm-norm', action='store_true')
        parser.add_argument('--set-diag', action='store_true')
        parser.add_argument('--remove-diag', action='store_true')

        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.directed,
            args.dropedge_rate,
            args.num_propagations,
            args.asymm_norm,
            args.set_diag,
            args.remove_diag,
        )

    def __init__(
        self,
        num_features,
        hidden_size,
        num_classes,
        num_layers,
        dropout,
        dropedge_rate,
        undirected,
        num_propagations,
        asymm_norm,
        set_diag,
        remove_diag,
    ):

        super(MLP, self).__init__()

        self.dropout = dropout
        self.dropedge_rate = dropedge_rate

        self.undirected = undirected
        self.num_propagations = num_propagations
        self.asymm_norm = asymm_norm
        self.set_diag = set_diag
        self.remove_diag = remove_diag

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear((1 + 2 * self.num_propagations) * num_features, hidden_size))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_size))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_size, hidden_size))
            self.bns.append(torch.nn.BatchNorm1d(hidden_size))
        self.lins.append(torch.nn.Linear(hidden_size, num_classes))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def _preprocessing(self, x, edge_index):
        num_nodes = x.shape[0]

        op_embedding = []
        op_embedding.append(x)

        # Convert to numpy arrays on cpu
        edge_index, _ = dropout_adj(edge_index, p=self.dropedge_rate, num_nodes=num_nodes)
        row, col = edge_index

        if self.undirected:
            edge_index = to_undirected(edge_index, num_nodes)
            row, col = edge_index

        # adj matrix
        adj = get_adj(
            row, col, num_nodes, asymm_norm=self.asymm_norm, set_diag=self.set_diag, remove_diag=self.remove_diag
        )

        nx = x
        for _ in range(self.num_propagations):
            nx = adj @ nx
            op_embedding.append(nx)

        # transpose adj matrix
        adj = get_adj(
            col, row, num_nodes, asymm_norm=self.asymm_norm, set_diag=self.set_diag, remove_diag=self.remove_diag
        )

        nx = x
        for _ in range(self.num_propagations):
            nx = adj @ nx
            op_embedding.append(nx)

        return torch.cat(op_embedding, dim=1)

    def forward(self, x, edge_index):
        x = self._preprocessing(x, edge_index)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)
