Tutorial
========

.. currentmodule:: CogDL

This guide can help you start working with CogDL.


Create a model
--------------

Here, we will create a spectral clustering model, which is a very simple graph embedding algorithm.
We name it spectral.py and put it in cogl/models/emb directory.

First we import necessary library like numpy, scipy, networkx, sklearn, we also import API like 'BaseModel' and 'register_model' from cogl/models/ to build our new model:

.. code-block:: python
    
    import numpy as np
    import networkx as nx
    import scipy.sparse as sp
    from sklearn import preprocessing
    from .. import BaseModel, register_model
  

Then we use function decorator to declare new model for CogDL

.. code-block:: python

        @register_model('spectral')
        class Spectral(BaseModel):
            (...)


We have to implement method 'build_model_from_args' in spectral.py. If it need more parameters to train, we can use 'add_args' to add model-specific arguments.

.. code-block:: python

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


Each new model should provide a 'train' method to obtain representation.

.. code-block:: python

    def train(self, G):
        matrix = nx.normalized_laplacian_matrix(G).todense()
        matrix = np.eye(matrix.shape[0]) - np.asarray(matrix)
        ut, s, _ = sp.linalg.svds(matrix, self.dimension)
        emb_matrix = ut * np.sqrt(s)
        emb_matrix = preprocessing.normalize(emb_matrix, "l2")
        return emb_matrix



Create a dataset
----------------

In order to add a dataset into CogDL, you should know your dataset's format. We have provided several graph format like edgelist, matlab_matrix and pyg.
If your dataset is same as the 'ppi' dataset, which contains two matrices: 'network' and 'group', you can register your dataset directly use above code.

.. code-block:: python

    @register_dataset("ppi")
    class PPIDataset(MatlabMatrix):
        def __init__(self):
            dataset, filename = "ppi", "Homo_sapiens"
            url = "http://snap.stanford.edu/node2vec/"
            path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
            super(PPIDataset, self).__init__(path, filename, url)

You should declare the name of the dataset, the name of file and the url, where our script can download resource.


Create a task
-------------
