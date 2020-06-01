import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from .. import BaseModel, register_model


@register_model("spectral")
class Spectral(BaseModel):
    r"""The Spectral clustering model from the `"Leveraging social media networks for classiﬁcation"
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.481.5392&rep=rep1&type=pdf>`_ paper
    
    Args:
        hidden_size (int) : The dimension of node representation.
    """    
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass 

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size)

    def __init__(self, dimension):
        super(Spectral, self).__init__()
        self.dimension = dimension

    def train(self, G):
        matrix = nx.normalized_laplacian_matrix(G).todense()
        matrix = np.eye(matrix.shape[0]) - np.asarray(matrix)
        ut, s, _ = sp.linalg.svds(matrix, self.dimension)
        emb_matrix = ut * np.sqrt(s)
        emb_matrix = preprocessing.normalize(emb_matrix, "l2")
        return emb_matrix

