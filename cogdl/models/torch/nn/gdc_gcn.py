import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import expm
from cogdl.layers import GCNLayer

from .. import BaseModel


class GDC_GCN(BaseModel):
    r"""The GDC model from the `"Diffusion Improves Graph Learning"
    <https://arxiv.org/abs/1911.05485>`_ paper, with the PPR and heat matrix variants
    combined with GCN

    Args:
        num_features (int)  : Number of input features in ppr-preprocessed dataset.
        num_classes (int)   : Number of classes.
        hidden_size (int)   : The dimension of node representation.
        dropout (float)     : Dropout rate for model training.
        alpha (float)       : PPR polynomial filter param, 0 to 1.
        t (float)           : Heat polynomial filter param
        k (int)             : Top k nodes retained during sparsification.
        eps (float)         : Threshold for clipping.
        gdc_type (str)      : "none", "ppr", "heat"
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        # GCN Params
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)

        # GDC Params
        parser.add_argument("--alpha", type=float, default=0.05)
        parser.add_argument("--t", type=float, default=5.0)
        parser.add_argument("--k", type=int, default=128)
        parser.add_argument("--eps", type=float, default=0.01)
        parser.add_argument("--gdc-type", default="ppr")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.dropout,
            args.alpha,
            args.t,
            args.k,
            args.eps,
            args.gdc_type,
        )

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, t, k, eps, gdctype):
        super(GDC_GCN, self).__init__()

        # preproc params
        self.alpha = alpha
        self.t = t
        self.k = k
        self.eps = eps
        self.gdc_type = gdctype

        self.data = None

        # GCN init
        self.nfeat = nfeat
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, graph):
        if self.data is None:
            self.reset_data(graph)
        graph = self.data
        x = graph.x
        if self.gdc_type == "none":
            graph.sym_norm()

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(graph, x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(graph, x)
        return x

    def predict(self, data=None):
        self.data.apply(lambda x: x.to(self.device))
        return self.forward(self.data)

    def reset_data(self, data):
        if self.data is None:
            data.to("cpu")
            self.data = self.preprocessing(data, gdc_type=self.gdc_type)
            data.to(self.device)

    def preprocessing(self, data, gdc_type="ppr"):
        # generate adjacency matrix from sparse representation

        def get_diffusion(x, edges):
            adj_matrix = self._get_adj_matrix(x, edges)

            if gdc_type == "none":
                print("No GDC filters chosen")
                processed_matrix = adj_matrix
            elif gdc_type == "ppr":
                print("PPR filters chosen")
                processed_matrix = self._get_ppr_matrix(adj_matrix, alpha=self.alpha)
            elif gdc_type == "heat":
                print("Heat filters chosen")
                processed_matrix = self._get_heat_matrix(adj_matrix, t=self.t)
            else:
                raise ValueError

            if gdc_type == "ppr" or gdc_type == "heat":
                if self.k:
                    print(f"Selecting top {self.k} edges per node.")
                    processed_matrix = self._get_top_k_matrix(processed_matrix, k=self.k)
                elif self.eps:
                    print(f"Selecting edges with weight greater than {self.eps}.")
                    processed_matrix = self._get_clipped_matrix(processed_matrix, eps=self.eps)
                else:
                    raise ValueError

            edges_i = []
            edges_j = []
            edge_attr = []
            for i, row in enumerate(processed_matrix):
                for j in np.where(row > 0)[0]:
                    edges_i.append(i)
                    edges_j.append(j)
                    edge_attr.append(processed_matrix[i, j])
            edge_index = [edges_i, edges_j]
            return torch.as_tensor(edge_index, dtype=torch.long), torch.as_tensor(edge_attr, dtype=torch.float)

        edge_index, edge_weight = get_diffusion(data.x, data.edge_index)
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        if self.training and data._adj_train is not None:
            data.eval()
            edge_index, edge_weight = get_diffusion(data.x, data.edge_index)
            data.edge_index = edge_index
            data.edge_weight = edge_weight
            data.train()
        return data

    def _get_adj_matrix(self, x, edge_index):
        num_nodes = x.shape[0]
        adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        return adj_matrix

    def _get_ppr_matrix(self, adj_matrix, alpha=0.1):
        num_nodes = adj_matrix.shape[0]
        A_tilde = adj_matrix + np.eye(num_nodes)
        D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
        H = D_tilde @ A_tilde @ D_tilde
        return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

    def _get_heat_matrix(self, adj_matrix, t=5.0):
        num_nodes = adj_matrix.shape[0]
        A_tilde = adj_matrix + np.eye(num_nodes)
        D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
        H = D_tilde @ A_tilde @ D_tilde
        return expm(-t * (np.eye(num_nodes) - H))

    def _get_top_k_matrix(self, A, k=128):
        num_nodes = A.shape[0]
        row_idx = np.arange(num_nodes)
        A[A.argsort(axis=0)[: num_nodes - k], row_idx] = 0.0
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1  # avoid dividing by zero
        return A / norm

    def _get_clipped_matrix(self, A, eps=0.01):
        A[A < eps] = 0.0
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1  # avoid dividing by zero
        return A / norm
