.. CogDL documentation master file

Welcome to CogDL's documentation!
=================================

CogDL is a graph representation learning toolkit that allows researchers and developers to easily train and compare baseline or custom models for node classification, link prediction and other tasks on graphs. It provides implementations of many popular models, including: non-GNN Baselines like Deepwalk, LINE, NetMF,  GNN Baselines like GCN, GAT, GraphSAGE.

.. image:: _static/cogdl-logo.png

CogDL provides these features:

- Sparsification: fast network embedding on large-scale networks with tens of millions of nodes
- Arbitrary: dealing with different graph structures: attributed, multiplex and heterogeneous networks
- Parallel: parallel training of different seeds and different models on multiple GPUs and automatically reporting the result table
- Extensible: easily register new datasets, models, criteria and tasks
- Supported tasks:
  - Node classification
  - Link prediction
  - Graph classification
  - Social influence
  - Community detection


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorial
   task/index
   license
   citing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
