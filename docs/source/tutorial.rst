Tutorial
========

.. currentmodule:: CogDL

This guide can help you start working with CogDL.


Create a model
--------------

Here, we will create a spectral clustering model, which is a very simple graph embedding algorithm.
We name it spectral.py and put it in cogdl/models/emb directory.

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

In order to evaluate some methods on several datasets, we can build a task and evaluate learned representation. The BaseTask class are: 

.. code-block:: python

    class BaseTask(object):
        @staticmethod
        def add_args(parser):
            """Add task-specific arguments to the parser."""
            pass

        def __init__(self, args):
            pass

        def train(self, num_epoch):
            raise NotImplementedError

we can create a subclass to implement 'train' method like CommunityDetection, which get representation of each node and apply clustering algorithm(K-means) to evaluate.

.. code-block:: python

    @register_task("community_detection")
    class CommunityDetection(BaseTask):
        """Community Detection task."""

        @staticmethod
        def add_args(parser):
            """Add task-specific arguments to the parser."""
            parser.add_argument("--hidden-size", type=int, default=128)
            parser.add_argument("--num-shuffle", type=int, default=5)

        def __init__(self, args):
            super(CommunityDetection, self).__init__(args)
            dataset = build_dataset(args)
            self.data = dataset[0]
  
            self.num_nodes, self.num_classes = self.data.y.shape
            self.label = np.argmax(self.data.y, axis=1)
            self.model = build_model(args)
            self.hidden_size = args.hidden_size
            self.num_shuffle = args.num_shuffle

        def train(self):
            G = nx.Graph()
            G.add_edges_from(self.data.edge_index.t().tolist())
            embeddings = self.model.train(G)

            clusters = [30, 50, 70]
            all_results = defaultdict(list)
            for num_cluster in clusters:
                for _ in range(self.num_shuffle):
                    model = KMeans(n_clusters=num_cluster).fit(embeddings)
                    nmi_score = normalized_mutual_info_score(self.label, model.labels_)
                    all_results[num_cluster].append(nmi_score)
                
            return dict(
                (
                    f"normalized_mutual_info_score {num_cluster}",
                    sum(all_results[num_cluster]) / len(all_results[num_cluster]),
                )
                for num_cluster in sorted(all_results.keys())
            )


Combine model, dataset and task
-------------------------------

After create your model, dataset and task, we could combine them together to learn representation from a model on a dataset and evaluate its performance according to a task.
We use 'build_model', 'build_dataset', 'build_task' method to build them with cooresponding parameters.

.. code-block:: python

    from cogdl.tasks import build_task
    from cogdl.datasets import build_dataset
    from cogdl.models import build_model
    from cogdl.utils import build_args_from_dict

    def test_deepwalk_ppi():
        default_dict = {'hidden_size': 64, 'num_shuffle': 1, 'cpu': True}
        args = build_args_from_dict(default_dict)
        
        # model, dataset and task parameters
        args.model = 'spectral'
        args.dataset = 'ppi'
        args.task = 'community_detection'
        
        # build model, dataset and task
        dataset = build_dataset(args)
        model = build_model(args)
        task = build_task(args)
        
        # train model and get evaluate results
        ret = task.train()
        print(ret)