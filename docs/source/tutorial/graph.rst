Graph Storage
=====================
A graph is used to store information of structured data. CogDL represents a graph with a ``cogdl.data.Graph`` object.
Briefly, a ``Graph`` holds the following attributes:

- ``x``: Node feature matrix with shape ``[num_nodes, num_features]``, `torch.Tensor`
- ``edge_index``:  COO format sparse matrix, `Tuple`
- ``edge_weight``: Edge weight with shape ``[num_edges,]``, `torch.Tensor`
- ``edge_attr``: Edge attribute matrix with shape ``[num_edges, num_attr]``
- ``y``: Target labels of each node, with shape ``[num_nodes,]`` for single label case and `[num_nodes, num_labels]` for mult-label case
- ``row_indptr``: Row index pointer for CSR sparse matrix, `torch.Tensor`.
- ``col_indices``: Column indices for CSR sparse matrix, `torch.Tensor`.
- ``num_nodes``: The number of nodes in graph.
- ``num_edges``: The number of edges in graph.

The above are the basic attributes but are not necessary. You may define a graph with `g = Graph(edge_index=edges)` and omit the others.
Besides, ``Graph`` is not restricted to these attributes and other self-defined attributes, e.x. `graph.mask = mask`, are also supported.

``Graph`` stores sparse matrix with COO or CSR format. COO format is easier to add or remove edges, e.x. `add_self_loops`, and CSR is stored for fast message-passing.
``Graph`` automatically convert between two formats and you can use both on demands without worrying. You can create a Graph with edges or assign edges
to a created graph. `edge_weight` will be automatically initialized as all ones, and you can modify it to fit your need.

.. code-block:: python

    import torch
    from cogdl.data import Graph
    edges = torch.tensor([[0,1],[1,3],[2,1],[4,2],[0,3]]).t()
    g = Graph()
    g.edge_index = edges
    g = Graph(edge_index=edges) # equivalent to that above
    print(g.edge_weight)
    >> tensor([1., 1., 1., 1., 1.])
    g.num_nodes
    >> 5
    g.num_edges
    >> 5
    g.edge_weight = torch.rand(5)
    print(g.edge_weight)
    >> tensor([0.8399, 0.6341, 0.3028, 0.0602, 0.7190])

We also implement commonly used operations in ``Graph``:

- ``add_self_loops``: add self loops for nodes in graph,

.. math::

    \hat{A}=A+I

- ``add_remaining_self_loops``: add self-loops for nodes without it.
- ``sym_norm``: symmetric normalization of edge_weight used `GCN`:

.. math::

    \hat{A}=D^{-1/2}AD^{-1/2}

- ``row_norm``: row-wise normalization of edge_weight:

.. math::

    \hat{A} = D^{-1}A

- ``degrees``: get degrees for each node. For directed graph, this function returns in-degrees of each node.

.. code-block:: python

    import torch
    from cogdl.data import Graph
    edge_index = torch.tensor([[0,1],[1,3],[2,1],[4,2],[0,3]]).t()
    g = Graph(edge_index=edge_index)
    >> Graph(edge_index=[2, 5])
    g.add_remaining_self_loops()
    >> Graph(edge_index=[2, 10], edge_weight=[10])
    >> print(edge_weight) # tensor([1., 1., ..., 1.])
    g.row_norm()
    >> print(edge_weight) # tensor([0.3333, ..., 0.50])

- ``subgraph``: get a subgraph containing given nodes and edges between them.
- ``edge_subgraph``: get a subgraph containing given edges and corresponding nodes.
- ``sample_adj``: sample a fixed number of neighbors for each given node.

.. code-block:: python

    from cogdl.datasets import build_dataset_from_name
    g = build_dataset_from_name("cora")[0]
    g.num_nodes
    >> 2707
    g.num_edges
    >> 10184
    # Get a subgraph contaning nodes [0, .., 99]
    sub_g = g.subgraph(torch.arange(100))
    >> Graph(x=[100, 1433], edge_index=[2, 18], y=[100])
    # Sample 3 neighbors for each nodes in [0, .., 99]
    nodes, adj_g = g.sample_adj(torch.arange(100), size=3)
    >> Graph(edge_index=[2, 300]) # adj_g

- ``train/eval``: In inductive settings, some nodes and edges are unseen during training, ``train/eval`` provides access to switching backend graph for training/evaluation. In transductive setting, you may ignore this.

.. code-block:: python

    # train_step
    model.train()
    graph.train()

    # inference_step
    model.eval()
    data.eval()



Mini-batch Graphs
--------------------

In node classification, all operations are in one single graph. But in tasks like graph classification, we need to deal with
many graphs with mini-batch. Datasets for graph classification contains graphs which can be accessed with index, e.x. ``data[2]``.
To support mini-batch training/inference, CogDL combines graphs in a batch into one whole graph, where adjacency matrices form sparse block diagnal matrices
and others(node features, labels) are concatenated in node dimension. ``cogdl.data.Dataloader`` handles the process.

.. code-block:: python

    from cogdl.data import DataLoader
    from cogdl.datasets import build_dataset_from_name

    dataset = build_dataset_from_name("mutag")
    >> MUTAGDataset(188)
    dataswet[0]
    >> Graph(x=[17, 7], y=[1], edge_index=[2, 38])
    loader = DataLoader(dataset, batch_size=8)
    for batch in loader:
        model(batch)
    >> Batch(x=[154, 7], y=[8], batch=[154], edge_index=[2, 338])




``batch`` is an additional attributes that indicate the respective graph the node belongs to. It is mainly used to do global
pooling, or called `readout` to generate graph-level representation. Concretely, ``batch`` is a tensor like:

.. math::

    batch=[0,..,0, 1,...,1, N-1,...,N-1]


The following code snippet shows how to do global pooling to sum over features of nodes in each graph:

.. code-block:: python

    def batch_sum_pooling(x, batch):
        batch_size = int(torch.max(batch.cpu())) + 1
        res = torch.zeros(batch_size, x.size(1)).to(x.device)
        out = res.scatter_add_(
            dim=0,
            index=batch.unsqueeze(-1).expand_as(x),
            src=x
           )
        return out



Editing Graphs
---------------
Mutation or changes can be applied to edges in some settings. In such cases, we need to `generate` a graph for calculation while
keep the original graph. CogDL provides `graph.local_graph` to set up a local scape and any out-of-place operation will not
reflect to the original graph. However, in-place operation will affect the original graph.


.. code-block:: python

    graph = build_dataset_from_name("cora")[0]
    graph.num_edges
    >> 10184
    with graph.local_graph():
        mask = torch.arange(100)
        row, col = graph.edge_index
        graph.edge_index = (row[mask], col[mask])
        graph.num_edges
        >> 100
    graph.num_edges
    >> 10184

    graph.edge_weight
    >> tensor([1.,...,1.])
    with graph.local_graph():
        graph.edge_weight += 1
    graph.edge_weight
    >> tensor([2.,...,2.])




Common benchmarks
-------------------

CogDL provides a bunch of commonly used datasets for graph tasks like node classification, graph classification and many others.
You can access them conveniently shown as follows. Statistics of datasets are on
this `page <https://github.com/THUDM/cogdl/blob/master/cogdl/datasets/README.md>`_ .

.. code-block:: python

    from cogdl.datasets import build_dataset_from_name, build_dataset
    dataset = build_dataset_from_name("cora")
    dataset = build_dataset(args) # args.dataet = "cora"



For all datasets for node classification, we use `train_mask`, `val_mask`, `test_mask` to denote
train/validation/test split for nodes.
