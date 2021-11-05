import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from .. import BaseModel


class GraRep(BaseModel):
    r"""The GraRep model from the `"Grarep: Learning graph representations with global structural information"
    <http://dl.acm.org/citation.cfm?doid=2806416.2806512>`_ paper.

    Args:
        hidden_size (int) : The dimension of node representation.
        step (int) : The maximum order of transitition probability.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--step", type=int, default=5,
                            help="Number of matrix step in GraRep. Default is 5.")
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.step)

    def __init__(self, dimension, step):
        super(GraRep, self).__init__()
        self.dimension = dimension
        self.step = step

    def forward(self, graph, return_dict=False):
        self.G = graph.to_networkx()
        self.num_node = self.G.number_of_nodes()
        A = np.asarray(nx.adjacency_matrix(self.G).todense(), dtype=float)
        A = preprocessing.normalize(A, "l1")

        log_beta = np.log(1.0 / self.num_node)
        A_list = [A]
        T_list = [sum(A).tolist()]
        temp = A
        # calculate A^1, A^2, ... , A^step, respectively
        for i in range(self.step - 1):
            temp = temp.dot(A)
            A_list.append(A)
            T_list.append(sum(temp).tolist())

        final_emb = np.zeros((self.num_node, 1))
        for k in range(self.step):
            for j in range(A.shape[1]):
                A_list[k][:, j] = np.log(A_list[k][:, j] / T_list[k][j] + 1e-20) - log_beta
                for i in range(A.shape[0]):
                    A_list[k][i, j] = max(A_list[k][i, j], 0)
            # concatenate all k-step representations
            if k == 0:
                dimension = self.dimension - int(self.dimension / self.step) * (self.step - 1)
                final_emb = self._get_embedding(A_list[k], dimension)
            else:
                W = self._get_embedding(A_list[k], self.dimension / self.step)
                final_emb = np.hstack((final_emb, W))

        embeddings = final_emb
        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(self.G.nodes()):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((graph.num_nodes, embeddings.shape[1]))
            nx_nodes = self.G.nodes()
            features_matrix[nx_nodes] = embeddings[np.arange(graph.num_nodes)]
        return features_matrix

    def _get_embedding(self, matrix, dimension):
        # get embedding from svd and process normalization for ut
        ut, s, _ = sp.linalg.svds(matrix, int(dimension))
        emb_matrix = ut * np.sqrt(s)
        emb_matrix = preprocessing.normalize(emb_matrix, "l2")
        return emb_matrix
