import numpy as np
import scipy.sparse as sp
from scipy.special import iv
import networkx as nx
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

from cogdl.utils.prone_utils import get_embedding_dense
from cogdl.data import Graph
from .. import BaseModel


class ProNE(BaseModel):
    r"""The ProNE model from the `"ProNE: Fast and Scalable Network Representation Learning"
    <https://www.ijcai.org/Proceedings/2019/0594.pdf>`_ paper.

    Args:
        hidden_size (int) : The dimension of node representation.
        step (int) : The number of items in the chebyshev expansion.
        mu (float) : Parameter in ProNE.
        theta (float) : Parameter in ProNE.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--step", type=int, default=5,
                            help="Number of items in the chebyshev expansion")
        parser.add_argument("--mu", type=float, default=0.2)
        parser.add_argument("--theta", type=float, default=0.5)
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.step, args.mu, args.theta)

    def __init__(self, dimension, step, mu, theta):
        super(ProNE, self).__init__()
        self.dimension = dimension
        self.step = step
        self.mu = mu
        self.theta = theta

    def forward(self, graph: Graph, return_dict=False):
        nx_g = graph.to_networkx()
        self.matrix0 = sp.csr_matrix(nx.adjacency_matrix(nx_g))

        features_matrix = self._pre_factorization(self.matrix0, self.matrix0)

        embeddings_matrix = self._chebyshev_gaussian(self.matrix0, features_matrix, self.step, self.mu, self.theta)

        embeddings = embeddings_matrix

        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(nx_g.nodes()):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((graph.num_nodes, embeddings.shape[1]))
            nx_nodes = nx_g.nodes()
            features_matrix[nx_nodes] = embeddings[np.arange(graph.num_nodes)]
        return features_matrix

    def _get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        smat = sp.csc_matrix(matrix)  # convert to sparse CSC format
        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    def _pre_factorization(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = sp.diags(neg, format="csr")
        neg = mask.dot(neg)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self._get_embedding_rand(F)
        return features_matrix

    def _chebyshev_gaussian(self, A, a, order=5, mu=0.5, s=0.2, plus=False, nn=False):
        # NE Enhancement via Spectral Propagation
        num_node = a.shape[0]

        if order == 1:
            return a

        A = sp.eye(num_node) + A
        DA = preprocessing.normalize(A, norm="l1")
        L = sp.eye(num_node) - DA

        M = L - mu * sp.eye(num_node)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
        emb = mm = conv
        if not plus:
            mm = A.dot(a - conv)
        if not nn:
            emb = get_embedding_dense(mm, self.dimension)
        return emb
