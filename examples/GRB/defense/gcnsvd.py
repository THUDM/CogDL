import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GCNLayer
from cogdl.models import BaseModel
import cogdl.utils.grb_utils as utils
import types


class GCNSVD(BaseModel):
    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument("--num-features", type=int)
    #     parser.add_argument("--num-classes", type=int)
    #     parser.add_argument("--num-layers", type=int, default=2)
    #     parser.add_argument("--hidden-size", type=int, default=64)
    #     parser.add_argument("--dropout", type=float, default=0.5)
    #     parser.add_argument("--residual", action="store_true")
    #     parser.add_argument("--norm", type=str, default=None)
    #     parser.add_argument("--feat-norm", type=types.FunctionType, default=None)
    #     parser.add_argument("--adj-norm", type=types.FunctionType, default=None)
    #     parser.add_argument("--activation", type=str, default="relu")
    #     parser.add_argument("--k", type=int, default=50)
    #     # fmt: on

    # @classmethod
    # def build_model_from_args(cls, args):
    #     return cls(
    #         args.num_features,
    #         args.hidden_size,
    #         args.num_classes,
    #         args.num_layers,
    #         args.dropout,
    #         args.activation,
    #         args.residual,
    #         args.norm,
    #         args.feat_norm,
    #         args.adj_norm,
    #         args.k,
    #     )

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout=0.0,
        activation=F.relu,
        residual=False,
        norm=None,
        feat_norm=None,
        adj_norm_func=None,
        k=50,
    ):
        super(GCNSVD, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_size) is int:
            hidden_size = [hidden_size] * (num_layers - 1)
        elif type(hidden_size) is list or type(hidden_size) is tuple:
            assert len(hidden_size) == (num_layers - 1), "Incompatible sizes between hidden_size and n_layers."
        n_features = [in_feats] + hidden_size + [out_feats]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if norm == "layernorm" and i == 0:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(
                GCNLayer(
                    in_features=n_features[i],
                    out_features=n_features[i + 1],
                    activation=activation if i != num_layers - 1 else None,
                    residual=residual if i != num_layers - 1 else False,
                    dropout=dropout if i != num_layers - 1 else 0.0,
                    norm=norm if i != num_layers - 1 else None,
                )
            )
        self.k = k

    def forward(self, graph):
        adj, h = utils.getGRBGraph(graph)
        adj = self.truncatedSVD(graph, self.k)
        adj = utils.adj_preprocess(adj=adj, adj_norm_func=self.adj_norm_func, device=h.device)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                h = layer(h)
            else:
                h = layer(graph, h)

        return h

    def truncatedSVD(self, graph, k=50):
        # edge_index = adj._indices()
        edge_index = graph.edge_index
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        adj = sp.csr_matrix((np.ones(len(row)), (row, col)))
        if sp.issparse(adj):
            adj = adj.asfptype()
            U, S, V = sp.linalg.svds(adj, k=k)
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(adj)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            diag_S = np.diag(S)

        new_adj = U @ diag_S @ V
        new_adj = sp.csr_matrix(new_adj)

        return new_adj
