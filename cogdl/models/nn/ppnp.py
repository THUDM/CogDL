import torch
import torch.nn.functional as F

from .. import BaseModel
from cogdl.utils import spmm
from .mlp import MLP


class PPNP(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--propagation-type", type=str, default="appnp")
        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--num-iterations", type=int, default=10)  # only for appnp
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.propagation_type,
            args.alpha,
            args.num_iterations,
        )

    def __init__(self, nfeat, nhid, nclass, num_layers, dropout, propagation, alpha, niter, cache=True):
        super(PPNP, self).__init__()
        # GCN as a prediction and then apply the personalized page rank on the results
        self.nn = MLP(nfeat, nclass, nhid, num_layers, dropout)
        if propagation not in ("appnp", "ppnp"):
            print("Invalid propagation type, using default appnp")
            propagation = "appnp"

        self.propagation = propagation
        self.alpha = alpha
        self.niter = niter
        self.dropout = dropout
        self.vals = None  # speedup for ppnp
        self.use_cache = cache
        self.cache = dict()

    def forward(self, graph):
        def get_ready_format(input, edge_index, edge_attr=None):
            if isinstance(edge_index, tuple):
                edge_index = torch.stack(edge_index)
            if edge_attr is None:
                edge_attr = torch.ones(edge_index.shape[1]).float().to(input.device)
            adj = torch.sparse_coo_tensor(edge_index, edge_attr, (input.shape[0], input.shape[0]),).to(input.device)
            return adj

        x = graph.x
        graph.sym_norm()
        # get prediction
        x = F.dropout(x, p=self.dropout, training=self.training)
        local_preds = self.nn.forward(x)

        # apply personalized pagerank
        if self.propagation == "ppnp":
            if self.vals is None:
                self.vals = self.alpha * torch.inverse(
                    torch.eye(x.shape[0]).to(x.device)
                    - (1 - self.alpha) * get_ready_format(x, graph.edge_index, graph.edge_attr)
                )
            final_preds = F.dropout(self.vals) @ local_preds
        else:  # appnp
            preds = local_preds
            with graph.local_graph():
                graph.edge_weight = F.dropout(graph.edge_weight, p=self.dropout, training=self.training)
                graph.set_symmetric()
                for _ in range(self.niter):
                    new_features = spmm(graph, preds)
                    preds = (1 - self.alpha) * new_features + self.alpha * local_preds
                final_preds = preds
        return final_preds

    def predict(self, graph):
        return self.forward(graph)
