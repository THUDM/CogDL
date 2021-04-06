import torch
import torch.nn.functional as F

from .. import BaseModel, register_model
from cogdl.utils import (
    add_remaining_self_loops,
    remove_self_loops,
    row_normalization,
    symmetric_normalization,
    to_undirected,
    spmm,
    dropout_adj,
)


def get_adj(graph, asymm_norm=False, set_diag=True, remove_diag=False):
    if set_diag:
        graph.add_remaining_self_loops()
    elif remove_diag:
        graph.remove_self_loops()
    if asymm_norm:
        graph.row_norm()
    else:
        graph.sym_norm()
    return graph


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

        self.cache_x = None

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def _preprocessing(self, graph, x):
        op_embedding = []
        op_embedding.append(x)

        edge_index = graph.edge_index

        # Convert to numpy arrays on cpu
        edge_index, _ = dropout_adj(edge_index, drop_rate=self.dropedge_rate)

        # if self.undirected:
        #     edge_index = to_undirected(edge_index, num_nodes)

        graph = get_adj(graph, asymm_norm=self.asymm_norm, set_diag=self.set_diag, remove_diag=self.remove_diag)

        with graph.local_graph():
            graph.edge_index = edge_index
            for _ in range(self.num_propagations):
                x = spmm(graph, x)
                op_embedding.append(x)

        for _ in range(self.num_propagations):
            nx = spmm(graph, x)
            op_embedding.append(nx)

        return torch.cat(op_embedding, dim=1)

    def forward(self, graph):
        if self.cache_x is None:
            x = graph.x
            self.cache_x = self._preprocessing(graph, x)
        x = self.cache_x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
