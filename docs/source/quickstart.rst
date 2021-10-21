Quick Start
===========


API Usage
---------

You can run all kinds of experiments through CogDL APIs, especially ``experiment()``. You can also use your own datasets and models for experiments. A quickstart example can be found in the `quick_start.py <https://github.com/THUDM/cogdl/tree/master/examples/quick_start.py>`_. More examples are provided in the `examples/ <https://github.com/THUDM/cogdl/tree/master/examples/>`_. 


.. code-block:: python

    from cogdl import experiment

    # basic usage
    experiment(dataset="cora", model="gcn")

    # set other hyper-parameters
    experiment(dataset="cora", model="gcn", hidden_size=32, max_epoch=200)

    # run over multiple models on different seeds
    experiment(dataset="cora", model=["gcn", "gat"], seed=[1, 2])

    # automl usage
    def search_space(trial):
        return {
            "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "dropout": trial.suggest_uniform("dropout", 0.5, 0.8),
        }

    experiment(dataset="cora", model="gcn", seed=[1, 2], search_space=search_space)

Command-Line Usage
------------------

You can also use ``python scripts/train.py --dataset example_dataset --model example_model`` to run example_model on example_data.

- ``--dataset``, dataset name to run, can be a list of datasets with space like ``cora citeseer``. Supported datasets include ``cora``, ``citeseer``, ``pumbed``, ``ppi``, ``flickr``. More datasets can be found in the `cogdl/datasets <https://github.com/THUDM/cogdl/tree/master/cogdl/datasets>`_.
- ``--model``, model name to run, can be a list of models like ``gcn gat``. Supported models include ``gcn``, ``gat``, ``graphsage``. More models can be found in the `cogdl/models <https://github.com/THUDM/cogdl/tree/master/cogdl/models>`_.

For example, if you want to run GCN and GAT on the Cora dataset, with 5 different seeds:

```bash
python scripts/train.py --dataset cora --model gcn gat --seed 0 1 2 3 4
```

Expected output:

===================  ==============  ===============
Variant              test_acc        val_acc        
===================  ==============  ===============
('cora', 'gcn')      0.8050±0.0047   0.7940±0.0063  
('cora', 'gat')      0.8234±0.0042   0.8088±0.0016  
===================  ==============  ===============


If you want to run parallel experiments on your server with multiple GPUs on multiple models/datasets:

.. code-block:: bash

    python scripts/parallel_train.py --dataset cora citeseer --model gcn gat --devices 0 1 --seed 0 1 2 3 4

Expected output:

====================  ==============  ===============
Variant               test_acc        val_acc        
====================  ==============  ===============
('cora', 'gcn')       0.8050±0.0047   0.7940±0.0063  
('cora', 'gat')       0.8234±0.0042   0.8088±0.0016  
('citeseer', 'gcn')   0.6938±0.0133   0.7108±0.0148
('citeseer', 'gat')   0.7098±0.0053   0.7244±0.0039  
====================  ==============  ===============

