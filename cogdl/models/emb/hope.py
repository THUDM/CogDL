import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from .. import BaseModel, register_model


@register_model("hope")
class HOPE(BaseModel):
    r"""The HOPE model from the `"Grarep: Asymmetric transitivity preserving graph embedding"
    <http://dl.acm.org/citation.cfm?doid=2939672.2939751>`_ paper.
    
    Args:
        hidden_size (int) : The dimension of node representation.
        beta (float) : Parameter in katz decomposition.
    """
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beta', type=float, default=0.01,
                            help='Parameter of katz for HOPE. Default is 0.01')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.beta)

    def __init__(self, dimension, beta):
        super(HOPE, self).__init__()
        self.dimension = dimension
        self.beta = beta

    def train(self, G):
        r"""The author claim that Katz has superior performance in related tasks
        S_katz = (M_g)^-1 * M_l = (I - beta*A)^-1 * beta*A = (I - beta*A)^-1 * (I - (I -beta*A))
        = (I - beta*A)^-1 - I
        """
        self.G = G
        adj = nx.adjacency_matrix(self.G).todense()
        n = adj.shape[0]
        katz_matrix = np.asarray((np.eye(n) - self.beta * np.mat(adj)).I - np.eye(n))
        self.embeddings = self._get_embedding(katz_matrix, self.dimension)
        return self.embeddings

    def _get_embedding(self, matrix, dimension):
        # get embedding from svd and process normalization for ut and vt
        ut, s, vt = sp.linalg.svds(matrix, int(dimension / 2))
        emb_matrix_1, emb_matrix_2 = ut, vt.transpose()

        emb_matrix_1 = emb_matrix_1 * np.sqrt(s)
        emb_matrix_2 = emb_matrix_2 * np.sqrt(s)
        emb_matrix_1 = preprocessing.normalize(emb_matrix_1, "l2")
        emb_matrix_2 = preprocessing.normalize(emb_matrix_2, "l2")
        features = np.hstack((emb_matrix_1, emb_matrix_2))
        return features
