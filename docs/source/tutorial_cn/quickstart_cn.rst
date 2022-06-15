快速开始
===========

API 用法
------------

您可以通过 CogDL 的 API 运行各种实验，尤其是experiment(). 您还可以使用自己的数据集和模型进行实验。快速入门的示例可以在
`quick_start.py <https://github.com/THUDM/cogdl/tree/master/examples/quick_start.py>`_.中找到。
`examples/ <https://github.com/THUDM/cogdl/tree/master/examples/>`_ 中提供了更多的示例。

.. code-block:: python

    from cogdl import experiment

    # basic usage
    experiment(dataset="cora", model="gcn")

    # set other hyper-parameters
    experiment(dataset="cora", model="gcn", hidden_size=32, epochs=200)

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

命令行用法
------------------

您可以使用命令 ``python scripts/train.py --dataset example_dataset --model example_model`` 运行 example_model 在example_data上.

- ``--dataset``,是要使用的数据集名称, 可以是带空格的数据集列表比如 ``cora citeseer``. 支持的数据集包括 ``cora``, ``citeseer``, ``pumbed``, ``ppi``, ``flickr`` 等等. 查看更多的数据集 `cogdl/datasets <https://github.com/THUDM/cogdl/tree/master/cogdl/datasets>`_
- ``--model``, 是要使用的模型名称, 可以是带空格的数据集列表比如 ``gcn gat``. 支持的模型包括 ``gcn``, ``gat``, ``graphsage`` 等等. 查看更多的模型 `cogdl/models <https://github.com/THUDM/cogdl/tree/master/cogdl/models>`_.


例如，如果你想在 Cora 数据集上运行 GCN 和 GAT，使用 5 个不同的seeds：

```bash
python scripts/train.py --dataset cora --model gcn gat --seed 0 1 2 3 4
```

预期结果:

===================  ==============  ===============
Variant              test_acc        val_acc
===================  ==============  ===============
('cora', 'gcn')      0.8050±0.0047   0.7940±0.0063
('cora', 'gat')      0.8234±0.0042   0.8088±0.0016
===================  ==============  ===============

如果您想在多个模型/数据集上使用多个 GPU 在您的服务器上并行的进行实验：

.. code-block:: bash

    python scripts/train.py --dataset cora citeseer --model gcn gat --devices 0 1 --seed 0 1 2 3 4

预期输出:

====================  ==============  ===============
Variant               test_acc        val_acc
====================  ==============  ===============
('cora', 'gcn')       0.8050±0.0047   0.7940±0.0063
('cora', 'gat')       0.8234±0.0042   0.8088±0.0016
('citeseer', 'gcn')   0.6938±0.0133   0.7108±0.0148
('citeseer', 'gat')   0.7098±0.0053   0.7244±0.0039
====================  ==============  ===============