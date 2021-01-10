import numpy as np
import torch
import torch.nn.functional as F
from cogdl.data import Data
from cogdl.utils import add_remaining_self_loops, symmetric_normalization
from scipy.linalg import expm

from .. import BaseModel, register_model
from .gcn import GraphConvolution


@register_model("gdc_gcn")
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
        gdc_type (str)            : "none", "ppr", "heat"
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
        parser.add_argument("--dataset", default=None)
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
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        edge_index, edge_attr = add_remaining_self_loops(edge_index)
        edge_attr = symmetric_normalization(x.shape[0], edge_index, edge_attr)
        adj_values = edge_attr

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, edge_index, adj_values))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, adj_values)
        return x

    def node_classification_loss(self, data):
        if self.data is None:
            self.reset_data(data)
        pred = self.forward(self.data.x, self.data.edge_index)
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(
            pred[self.data.train_mask],
            self.data.y[self.data.train_mask],
        )

    def predict(self, data=None):
        return self.forward(data.x, data.edge_index)

    def reset_data(self, data):
        if self.data is None:
            data.apply(lambda x: x.cpu())
            self.data = self.preprocessing(data, gdc_type=self.gdc_type)
            data.apply(lambda x: x.to(self.device))

    def preprocessing(self, data, gdc_type="ppr"):
        # generate adjacency matrix from sparse representation
        adj_matrix = self._get_adj_matrix(data.x, data.edge_index)

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

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(processed_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(processed_matrix[i, j])
        edge_index = [edges_i, edges_j]

        data = Data(
            x=data.x,
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=data.y,
            train_mask=data.train_mask,
            test_mask=data.test_mask,
            val_mask=data.val_mask,
        )
        data.apply(lambda x: x.to(self.device))

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
