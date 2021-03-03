Quick Start
===========


API Usage
---------

You can run all kinds of experiments through CogDL APIs, especially ``experiment()``. You can also use your own datasets and models for experiments. A quickstart example can be found in the `quick_start.py <https://github.com/THUDM/cogdl/tree/master/examples/quick_start.py>`_. More examples are provided in the `examples/ <https://github.com/THUDM/cogdl/tree/master/examples/>`_. 


.. code-block:: python

    from cogdl import experiment

    # basic usage
    experiment(task="node_classification", dataset="cora", model="gcn")

    # set other hyper-parameters
    experiment(task="node_classification", dataset="cora", model="gcn", hidden_size=32, max_epoch=200)

    # run over multiple models on different seeds
    experiment(task="node_classification", dataset="cora", model=["gcn", "gat"], seed=[1, 2])

    # automl usage
    def func_search(trial):
        return {
            "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "dropout": trial.suggest_uniform("dropout", 0.5, 0.8),
        }

    experiment(task="node_classification", dataset="cora", model="gcn", seed=[1, 2], func_search=func_search)

Command-Line Usage
------------------

You can also use ``python scripts/train.py --task example_task --dataset example_dataset --model example_model`` to run example_model on example_data and evaluate it via example_task.

- ``--task``, downstream tasks to evaluate representation like ``node_classification``, ``unsupervised_node_classification``, ``graph_classification``. More tasks can be found in the `cogdl/tasks <https://github.com/THUDM/cogdl/tree/master/cogdl/tasks>`_.
- ``--dataset``, dataset name to run, can be a list of datasets with space like ``cora citeseer ppi``. Supported datasets include 'cora', 'citeseer', 'pumbed', 'ppi', 'wikipedia', 'blogcatalog', 'flickr'. More datasets can be found in the `cogdl/datasets <https://github.com/THUDM/cogdl/tree/master/cogdl/datasets>`_.
- ``--model``, model name to run, can be a list of models like ``deepwalk line prone``. Supported models include 'gcn', 'gat', 'graphsage', 'deepwalk', 'node2vec', 'hope', 'grarep', 'netmf', 'netsmf', 'prone'. More models can be found in the `cogdl/models <https://github.com/THUDM/cogdl/tree/master/cogdl/models>`_.

For example, if you want to run LINE, NetMF on Wikipedia with unsupervised node classification task, with 5 different seeds:

.. code-block:: bash

    python scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model line netmf --seed 0 1 2 3 4


Expected output:

=========================  ==============  ==============  ==============  ==============  ============== 
Variant                    Micro-F1 0.1    Micro-F1 0.3    Micro-F1 0.5    Micro-F1 0.7    Micro-F1 0.9
=========================  ==============  ==============  ==============  ==============  ============== 
('wikipedia', 'line')      0.4069±0.0011   0.4071±0.0010   0.4055±0.0013   0.4054±0.0020   0.4080±0.0042
('wikipedia', 'netmf')     0.4551±0.0024   0.4932±0.0022   0.5046±0.0017   0.5084±0.0057   0.5125±0.0035
=========================  ==============  ==============  ==============  ==============  ============== 


If you want to run parallel experiments on your server with multiple GPUs on multiple models, GCN and GAT, on the Cora dataset with node classification task:

.. code-block:: bash

    python scripts/parallel_train.py --task node_classification --dataset cora --model gcn gat --device-id 0 1 --seed 0 1 2 3 4


Expected output:

=========================  ============== 
Variant                    Acc   
=========================  ============== 
('cora', 'gcn')            0.8236±0.0033  
('cora', 'gat')            0.8262±0.0032  
=========================  ============== 


Fast-Spmm Usage
---------------

CogDL provides a fast sparse matrix-matrix multiplication operator called `GE-SpMM <https://arxiv.org/abs/2007.03179>`_ to speed up training of GNN models on the GPU. 
You can set ``fast_spmm=True`` in the API usage or ``--fast-spmm`` in the command-line usage to enable this feature.
Note that this feature is still in testing and may not work under some versions of CUDA.
