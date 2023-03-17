import os

import torch

from .. import BaseModel
from .mlp import MLP
from cogdl.utils import spmm, dropout_adj, to_undirected


def get_adj(graph, remove_diag=False):
    if remove_diag:
        graph.remove_self_loops()
    else:
        graph.add_remaining_self_loops()
    return graph


def multi_hop_sgc(graph, x, nhop):
    results = []
    for _ in range(nhop):
        x = spmm(graph, x)
        results.append(x)
    return results


def multi_hop_ppr_diffusion(graph, x, nhop, alpha=0.5):
    results = []
    for _ in range(nhop):
        x = (1 - alpha) * x + spmm(graph, x)
        results.append(x)
    return results


class SIGN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        MLP.add_args(parser)
        parser.add_argument("--dropedge-rate", type=float, default=0.2)
        parser.add_argument("--directed", action="store_true")
        parser.add_argument("--nhop", type=int, default=3)
        parser.add_argument("--adj-norm", type=str, default=["sym"], nargs="+")
        parser.add_argument("--remove-diag", action="store_true")
        parser.add_argument("--diffusion", type=str, default="ppr")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        cls.dataset_name = args.dataset if hasattr(args, "dataset") else None
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.dropedge_rate,
            args.nhop,
            args.adj_norm,
            args.diffusion,
            args.remove_diag,
            not args.directed,
            args.norm,
            args.activation,
        )

    def __init__(
        self,
        num_features,
        hidden_size,
        num_classes,
        num_layers,
        dropout,
        dropedge_rate,
        nhop,
        adj_norm,
        diffusion="ppr",
        remove_diag=False,
        undirected=True,
        norm="batchnorm",
        activation="relu",
    ):
        super(SIGN, self).__init__()
        self.dropedge_rate = dropedge_rate

        self.undirected = undirected
        self.num_propagations = nhop
        self.adj_norm = adj_norm
        self.remove_diag = remove_diag
        self.diffusion = diffusion

        num_features = num_features * (1 + nhop * len(adj_norm))
        self.mlp = MLP(
            in_feats=num_features,
            out_feats=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm=norm,
        )

        self.cache_x = None

    def _preprocessing(self, graph, x, drop_edge=False):
        device = x.device
        graph.to("cpu")
        x = x.to("cpu")

        graph.eval()

        op_embedding = [x]

        edge_index = graph.edge_index
        if self.undirected:
            edge_index = to_undirected(edge_index)

        if drop_edge:
            edge_index, _ = dropout_adj(edge_index, drop_rate=self.dropedge_rate)

        graph = get_adj(graph, remove_diag=self.remove_diag)

        for norm in self.adj_norm:
            with graph.local_graph():
                graph.edge_index = edge_index
                graph.normalize(norm)
                if self.diffusion == "ppr":
                    results = multi_hop_ppr_diffusion(graph, graph.x, self.num_propagations)
                else:
                    results = multi_hop_sgc(graph, graph.x, self.num_propagations)
                op_embedding.extend(results)

        graph.to(device)
        return torch.cat(op_embedding, dim=1).to(device)

    def preprocessing(self, graph, x):
        print("Preprocessing...")
        dataset_name = None
        if self.dataset_name is not None:
            adj_norm = ",".join(self.adj_norm)
            dataset_name = f"{self.dataset_name}_{self.num_propagations}_{self.diffusion}_{adj_norm}.pt"
            if os.path.exists(dataset_name):
                return torch.load(dataset_name).to(x.device)
        if graph.is_inductive():
            graph.train()
            x_train = self._preprocessing(graph, x, drop_edge=True)
            graph.eval()
            x_all = self._preprocessing(graph, x, drop_edge=False)
            train_nid = graph.train_nid
            x_all[train_nid] = x_train[train_nid]
        else:
            x_all = self._preprocessing(graph, x, drop_edge=False)

        if dataset_name is not None:
            torch.save(x_all.cpu(), dataset_name)
        print("Preprocessing Done...")
        return x_all

    def forward(self, graph):
        if self.cache_x is None:
            x = graph.x.contiguous()
            self.cache_x = self.preprocessing(graph, x)
        x = self.cache_x
        x = self.mlp(x)
        return x
