Using customized Dataset
=========================

CogDL has provided lots of common datasets. But you may wish to apply GNN to new datasets for different applications. CogDL
provides an interface for customized datasets. You take care of reading in the dataset and the rest is to CogDL

We provide ``NodeDataset`` and ``GraphDataset`` as abstract classes and implement necessary basic operations.

Dataset for node_classification
---------------------------------
To create a dataset for node_classification, you need to inherit ``NodeDataset``. ``NodeDataset`` is for node-level prediction. Then you need to implement ``process`` method.
In this method, you are expected to read in your data and preprocess raw data to the format available to CogDL with ``Graph``.
Afterwards, we suggest you to save the processed data (we will also help you do it as you return the data) to avoid doing
the preprocessing again. Next time you run the code, CogDL will directly load it.

The running process of the module is as follows:

1. Specify the path to save processed data with `self.path`
2. Function `process` is called to load and preprocess data and your data is saved as `Graph` in `self.path`. This step
will be implemented the first time you use your dataset. And then every time you use your dataset, the dataset will be
loaded from `self.path` for convenience.
3. For dataset, for example, named `MyNodeDataset` in node-level tasks, You can access the data/Graph via
`MyNodeDataset.data` or `MyDataset[0]`.

In addition, evaluation metric for your dataset should be specified. CogDL provides ``accuracy`` and ``multiclass_f1``
for multi-class classification, ``multilabel_f1`` for multi-label classification.

If ``scale_feat`` is set to be `True`, CogDL will normalize node features with mean `u` and variance `s`:

.. math::

    z = (x - u) / s


Here is an `example <https://github.com/THUDM/cogdl/blob/master/examples/custom_dataset.py>`_:


.. code-block:: python

    from cogdl.data import Graph
    from cogdl.datasets import NodeDataset, generate_random_graph

    class MyNodeDataset(NodeDataset):
        def __init__(self, path="data.pt"):
            self.path = path
            super(MyNodeDataset, self).__init__(path, scale_feat=False, metric="accuracy")

        def process(self):
            """You need to load your dataset and transform to `Graph`"""
            num_nodes, num_edges, feat_dim = 100, 300, 30

            # load or generate your dataset
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            x = torch.randn(num_nodes, feat_dim)
            y = torch.randint(0, 2, (num_nodes,))

            # set train/val/test mask in node_classification task
            train_mask = torch.zeros(num_nodes).bool()
            train_mask[0 : int(0.3 * num_nodes)] = True
            val_mask = torch.zeros(num_nodes).bool()
            val_mask[int(0.3 * num_nodes) : int(0.7 * num_nodes)] = True
            test_mask = torch.zeros(num_nodes).bool()
            test_mask[int(0.7 * num_nodes) :] = True
            data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
            return data

    if __name__ == "__main__":
        # Train customized dataset via defining a new class
        dataset = MyNodeDataset()
        experiment(dataset=dataset, model="gcn")

        # Train customized dataset via feeding the graph data to NodeDataset
        data = generate_random_graph(num_nodes=100, num_edges=300, num_feats=30)
        dataset = NodeDataset(data=data)
        experiment(dataset=dataset, model="gcn")



Dataset for graph_classification
----------------------------------
Similarly, you need to inherit ``GraphDataset`` when you want to build a dataset for graph-level tasks such as `graph_classification`.
The overall implementation is similar while the difference is in ``process``. As ``GraphDataset`` contains a lot of graphs,
you need to transform your data to ``Graph`` for each graph separately to form a list of ``Graph``.
An example is shown as follows:

.. code-block:: python

    from cogdl.data import Graph
    from cogdl.datasets import GraphDataset

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
            return graphs

    if __name__ == "__main__":
        dataset = MyGraphDataset()
        experiment(model="gin", dataset=dataset)
