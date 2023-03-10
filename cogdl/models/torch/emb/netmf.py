import networkx as nx
import numpy as np
import scipy.sparse as sp

from .. import BaseModel


class NetMF(BaseModel):
    r"""The NetMF model from the `"Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec"
    <http://arxiv.org/abs/1710.02971>`_ paper.

    Args:
        hidden_size (int) : The dimension of node representation.
        window_size (int) : The actual context size which is considered in language model.
        rank (int) : The rank in approximate normalized laplacian.
        negative (int) : The number of nagative samples in negative sampling.
        is-large (bool) : When window size is large, use approximated deepwalk matrix to decompose.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--window-size", type=int, default=5)
        parser.add_argument("--rank", type=int, default=256)
        parser.add_argument("--negative", type=int, default=1)
        parser.add_argument("--is-large", action="store_true")
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.window_size, args.rank, args.negative, args.is_large)

    def __init__(self, dimension, window_size, rank, negative, is_large=False):
        super(NetMF, self).__init__()
        self.dimension = dimension
        self.window_size = window_size
        self.rank = rank
        self.negative = negative
        self.is_large = is_large

    def forward(self, graph, return_dict=False):
        nx_g = graph.to_networkx()
        A = sp.csr_matrix(nx.adjacency_matrix(nx_g))
        if not self.is_large:
            print("Running NetMF for a small window size...")
            deepwalk_matrix = self._compute_deepwalk_matrix(A, window=self.window_size, b=self.negative)

        else:
            print("Running NetMF for a large window size...")
            vol = float(A.sum())
            evals, D_rt_invU = self._approximate_normalized_laplacian(A, rank=self.rank, which="LA")
            deepwalk_matrix = self._approximate_deepwalk_matrix(
                evals, D_rt_invU, window=self.window_size, vol=vol, b=self.negative
            )
        # factorize deepwalk matrix with SVD
        u, s, _ = sp.linalg.svds(deepwalk_matrix, self.dimension)
        embeddings = sp.diags(np.sqrt(s)).dot(u.T).T

        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(nx_g.nodes()):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((graph.num_nodes, embeddings.shape[1]))
            nx_nodes = nx_g.nodes()
            features_matrix[nx_nodes] = embeddings[np.arange(graph.num_nodes)]
        return features_matrix

    def _compute_deepwalk_matrix(self, A, window, b):
        # directly compute deepwalk matrix
        n = A.shape[0]
        vol = float(A.sum())
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} A D^{-1/2}
        X = sp.identity(n) - L
        S = np.zeros_like(X)
        X_power = sp.identity(n)
        for i in range(window):
            print("Compute matrix %d-th power", i + 1)
            X_power = X_power.dot(X)
            S += X_power
        S *= vol / window / b
        D_rt_inv = sp.diags(d_rt ** -1)
        M = D_rt_inv.dot(D_rt_inv.dot(S).T).todense()
        M[M <= 1] = 1
        Y = np.log(M)
        return sp.csr_matrix(Y)

    def _approximate_normalized_laplacian(self, A, rank, which="LA"):
        # perform eigen-decomposition of D^{-1/2} A D^{-1/2} and keep top rank eigenpairs
        n = A.shape[0]
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} W D^{-1/2}
        X = sp.identity(n) - L
        print("Eigen decomposition...")
        evals, evecs = sp.linalg.eigsh(X, rank, which=which)
        print("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
        print("Computing D^{-1/2}U..")
        D_rt_inv = sp.diags(d_rt ** -1)
        D_rt_invU = D_rt_inv.dot(evecs)
        return evals, D_rt_invU

    def _deepwalk_filter(self, evals, window):
        for i in range(len(evals)):
            x = evals[i]
            evals[i] = 1.0 if x >= 1 else x * (1 - x ** window) / (1 - x) / window
        evals = np.maximum(evals, 0)
        print(
            "After filtering, max eigenvalue=%f, min eigenvalue=%f", np.max(evals), np.min(evals),
        )
        return evals

    def _approximate_deepwalk_matrix(self, evals, D_rt_invU, window, vol, b):
        # approximate deepwalk matrix
        evals = self._deepwalk_filter(evals, window=window)
        X = sp.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
        M = X.dot(X.T) * vol / b
        M[M <= 1] = 1
        Y = np.log(M)
        print("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(Y))
        return sp.csr_matrix(Y)
