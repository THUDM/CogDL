Using customized Dataset
=========================

CogDL has provided lots of common datasets. But you may wish to apply GNN to new datasets for different applications. CogDL
provides an interface for customized datasets. You take care of reading in the dataset and the rest is to CogDL

We provide ``NodeDataset`` and ``GraphDataset`` as abstract classes and implement necessary basic operations.

Dataset for node_classification
---------------------------------
To create a dataset for node_classification, you need to inherit ``NodeDataset``. ``NodeDataset`` is for tasks like `node_classification`
or `unsupervised_node_classification`, which focus on node-level prediction. Then you need to implement ``process`` method.
In this method, you are expected to read in your data and preprocess raw data to the format available to CogDL with ``Graph``.
Afterwards, we suggest you to save the processed data (we will also help you do it as you return the data) to avoid doing
the preprocessing again. Next time you run the code, CogDL will directly load it.

In addition, evaluation metric for your dataset should be specified. CogDL provides ``accuracy`` and ``multiclass_f1``
for multi-class classification, ``multilabel_f1`` for multi-label classification.

If ``scale_feat`` is set to be `True`, CogDL will normalize node features with mean `u` and variance `s`:

.. math::

    z = (x - u) / s


Here is an example:


.. code-block:: python

    from cogdl.data import Graph
    from cogdl.datasets import NodeDataset, register_dataset

    @register_dataset("node_dataset")
    class MyNodeDataset(NodeDataset):
        def __init__(self, path="data.pt"):
            self.path = path
            super(MyNodeDataset, self).__init__(path, scale_feat=False, metric="accuracy")

        def process(self):
            """You need to load your dataset and transform to `Graph`"""
            # Load and preprocess data
            edge_index = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3]).t()
            x = torch.randn(4, 10)
            mask = torch.bool(4)
            # provide attributes as you need
            data = Graph(x=x, edge_index=edge_index)
            torch.save(data, self.path)
            return data




Dataset for graph_classification
----------------------------------
Similarly, you need to inherit ``GraphDataset`` when you want to build a dataset for graph-level tasks such as `graph_classification`.
The overall implementation is similar while the difference is in ``process``. As ``GraphDataset`` contains a lot of graphs,
you need to transform your data to ``Graph`` for each graph separately to form a list of ``Graph``.
An example is shown as follows:

.. code-block:: python

    from cogdl.datasets import GraphDataset

    @register_dataset("graph_dataset")
    class MyGraphDataset(GraphDataset):
        def __init__(self, path="data.pt"):
            self.path = path
            super(MyGraphDataset, self).__init__(path, metric="accuracy")

        def process(self):
            # Load and preprocess data
            # Here we randomly generate several graphs for simplicity as an example
            graphs = []
            for i in range(10):
                edges = torch.randint(0, 20, (2, 30))
                label = torch.randint(0, 7, (1,))
                graphs.append(Graph(edge_index=edges, y=label))
            torch.save(graphs, self.path)
            return graphs




Use custom dataset with CogDL
---------------------------------
Now that you have set up your dataset, you can use models/task in CogDL immediately to get results.

.. code-block:: python

    # Use the GCN model with the dataset we define above
    dataset = MyNodeDataset("data.pt")
    args.model = "gcn"
    task = build_task(args, dataset=dataset)
    task.train()

    # Or you may simple run the command after `register_dataset`
    experiment(model="gcn", task="node_classification", dataset="node_dataset")

    # That's the same for other tasks
    experiment(model="gin", task="graph_classification", dataset="graph_dataset")

