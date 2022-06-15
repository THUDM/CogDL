自定义数据集
=========================
CogDL 提供了很多常见的数据集。但是您可能希望将 GNN 使用在不同应用的新数据集。CogDL 为自定义数据集提供了一个接口。你负责读取数据集，剩下的就交给CogDL。

我们提供 ``NodeDataset`` and ``GraphDataset`` 作为抽象类并实现必要的基本操作

node_classification 的数据集
----------------------------------
要为 node_classification 创建数据集，您需要继承 ``NodeDataset``。 ``NodeDataset`` 用于节点级预测。然后你需要实现 ``process`` 方法。在这种方法中，您需要读入数据并将原始数据预处理为 CogDL 可用的格式 ``Graph``。
之后，我们建议您保存处理后的数据（我们也会在您返回数据时帮助您保存）以避免再次进行预处理。下次运行代码时，CogDL 将直接加载它。

该模块的运行过程如下：

1.用 `self.path` 指定保存处理数据的路径
2. 调用 `process` 函数过程来加载和预处理数据，并将您的数据保存为 `Graph` 在 `self.path` 中。此步骤将在您第一次使用数据集时实施。然后每次使用数据集时，为了方便起见，将从 `self.path` 中加载数据集。
3. 对于数据集，例如节点级任务中名为 `MyNodeDataset` 的数据集，您可以通过 `MyNodeDataset.data` 或  `MyDataset[0]` 访问data/Graph。

此外，应指定数据集评价指标。CogDL 提供 `accuracy` 和 `multiclass_f1` 用于多类别分类， `multilabel_f1` 用于多标签分类。

如果 ``scale_feat`` 设置为 `True` ，CogDL 将使用均值 u 和方差 s 对节点特征进行归一化：

.. math::

    z = (x - u) / s

这是一个 `例子 <https://github.com/THUDM/cogdl/blob/master/examples/custom_dataset.py>`_:

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

graph_classification的数据集
----------------------------------
当您要为图级别任务（例如 ``graph_classification`` ）构建数据集时，您需要继承 ``GraphDataset`` ，总体实现是相似的，而区别在于process. 由于 ``GraphDataset``
包含大量图，您需要将你的数据转换为 ``Graph`` 为每个图成 ``Graph`` 列表。 一个例子如下所示：

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