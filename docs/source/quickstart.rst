Quick Start
===========


API Usage
---------

You can run all kinds of experiments through CogDL APIs, especially `build_task`. You can also use your own datasets and models for experiments. Some examples are provided in the `examples/ <https://github.com/THUDM/cogdl/tree/master/examples/>`_, including `gcn.py <https://github.com/THUDM/cogdl/tree/master/examples/gcn.py>`_. 

.. code-block:: python

    >>> from cogdl.tasks import build_task
    >>> from cogdl.options import get_default_args

    >>> # Get default hyper-parameters for experiments
    >>> args = get_default_args(task="node_classification", dataset="cora", model="gcn")
    >>> # Build and run
    >>> task = build_task(args)
    >>> ret = task.train()

Command-Line Usage
------------------

You can use ``python scripts/train.py --task example_task --dataset example_dataset --model example_method`` to run example_method on example_data and evaluate it via example_task.

- ``--task``, downstream tasks to evaluate representation like node_classification, unsupervised_node_classification, link_prediction. More tasks can be found in the `cogdl/tasks <https://github.com/THUDM/cogdl/tree/master/cogdl/tasks>`_.
- ``--dataset``, dataset name to run, can be a list of datasets with space like ``cora citeseer ppi``. Supported datasets include 'cora', 'citeseer', 'pumbed', 'PPI', 'wikipedia', 'blogcatalog', 'flickr'. More datasets can be found in the `cogdl/datasets <https://github.com/THUDM/cogdl/tree/master/cogdl/datasets>`_.
- ``--model``, model name to run, can be a list of models like ``deepwalk line prone``. Supported models include 'gcn', 'gat', 'graphsage', 'deepwalk', 'node2vec', 'hope', 'grarep', 'netmf', 'netsmf', 'prone'. More models can be found in the `cogdl/models <https://github.com/THUDM/cogdl/tree/master/cogdl/models>`_.

For example, if you want to run Deepwalk, Line, Netmf on Wikipedia with node classification task, with 5 different seeds:


.. code-block:: bash

    >>> python scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model line netmf --seed 0 1 2 3 4


Expected output:

=========================  ==============  ==============  ==============  ==============  ============== 
Variant                    Micro-F1 0.1    Micro-F1 0.3    Micro-F1 0.5    Micro-F1 0.7    Micro-F1 0.9
=========================  ==============  ==============  ==============  ==============  ============== 
('wikipedia', 'line')      0.4069±0.0011   0.4071±0.0010   0.4055±0.0013   0.4054±0.0020   0.4080±0.0042
('wikipedia', 'netmf')     0.4551±0.0024   0.4932±0.0022   0.5046±0.0017   0.5084±0.0057   0.5125±0.0035
=========================  ==============  ==============  ==============  ==============  ============== 


If you want to run parallel experiments on your server with multiple GPUs on multiple models gcn, gat on multiple datasets Cora, Citeseer with node classification task:

.. code-block:: bash

    >>> python scripts/parallel_train.py --task node_classification --dataset cora --model gcn gat --device-id 0 1 --seed 0 1 2 3 4


Expected output:

=========================  ============== 
Variant                    Acc   
=========================  ============== 
('cora', 'gcn')            0.8236±0.0033  
('cora', 'gat')            0.8262±0.0032  
=========================  ============== 
