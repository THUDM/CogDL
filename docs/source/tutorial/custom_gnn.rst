Using customized GNN
=====================

Sometimes you would like to design your own GNN module or use GNN for other purposes. In this chapter, we introduce how to
use GNN layer in CogDL to write your own GNN model and how to write a GNN layer from scratch.


GNN layers in CogDL to Define model
--------------------------------------
CogDL has implemented popular GNN layers in ``cogdl.layers``, and they can serve as modules to help design new GNNs.
Here is how we implement `Jumping Knowledge Network <https://arxiv.org/abs/1806.03536>`_ (JKNet) with ``GCNLayer`` in CogDL.

JKNet collects the output of all layers and concatenate them together to get the result:

.. math::
    :nowrap:

    \begin{gather*}
    H^{(0)} = X \\
    H^{(i+1)} = \sigma(\hat{A} H^{(i)} W^{(i)} \\
    OUT = CONCAT([H^{(0)},...,H^{(L)})
    \end{gather*}



.. code-block:: python

    import torch
    from cogdl.models import register_model

    @register_model("jknet")
    class JKNet(BaseModel):
        def __init__(self, in_feats, out_feats, hidden_size, num_layers):
            super(JKNet, self).__init__()
            shapes = [in_feats] + [hidden_size] * num_layers
            #
            self.layers = nn.ModuleList([
                GCNLayer(shape[i], shape[i+1])
                for i in range(num_layers)
            ])
            self.fc = nn.Linear(hidden_size * num_layers, out_feats)

        def forward(self, graph):
            graph.add_remaining_self_loops()
            graph.sym_norm()
            h = graph.x
            out = []
            for layer in self.layers:
                h = layer(x)
                out.append(h)
            out = torch.cat(out, dim=1)
            return self.fc(out)




Define your GNN Module
-----------------------
In most cases, you may build a layer module with new message propagation and aggragation scheme. Here the code snippet
shows how to implement a GCNLayer using ``Graph`` and efficient sparse matrix operators in CogDL.

.. code-block:: python

    import torch
    from cogdl.utils import spmm

    class GCNLayer(torch.nn.Module):
        """
        Args:
            in_feats: int
                Input feature size
            out_feats: int
                Output feature size
        """
        def __init__(self, in_feats, out_feats):
            super(GCNLayer, self).__init__()
            self.fc = torch.nn.Linear(in_feats, out_feats)

        def forward(self, graph, x):
            # symmetric normalization of adjacency matrix
            graph.sym_norm()
            h = self.fc(x)
            h = spmm(graph, h)
            return h




``spmm`` is sparse matrix multiplication operation frequently used in GNNs.

.. math::

        H = AH = SpMM(A, H)


Sparse matrix is stored  in ``Graph`` and will be called automatically. Message-passing in spatial space is equivalent to
matrix operations. CogDL also supports other efficient operators like ``edge_softmax`` and ``multi_head_spmm``, you can refer
to this `page <https://github.com/THUDM/cogdl/blob/master/cogdl/models/nn/gat.py>`_ for usage.


Use Custom models with CogDL
-----------------------------
Now that you have defined your own GNN, you can use dataset/task in CogDL to immediately train and evaluate the performance of your model.


.. code-block:: python

    data = dataset.data
    # Use the JKNet model as defined above
    model = JKNet(data.num_features, data.num_classes, 32, 4)
    task = build_task(args, dataset=dataset, model=model)
    task.train()

    # Or you may simple run the command after `register_model`
    experiment(model="jknet", task="node_classification", dataset="cora")

