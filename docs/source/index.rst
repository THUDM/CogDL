.. CogDL documentation master file

Welcome to CogDL's Documentation!
=================================

.. image:: _static/cogdl-logo.png

CogDL is a graph representation learning toolkit that allows researchers and developers to easily train and compare baseline or custom models for node classification, link prediction and other tasks on graphs. It provides implementations of many popular models, including: non-GNN Baselines like Deepwalk, LINE, NetMF, GNN Baselines like GCN, GAT, GraphSAGE.

CogDL provides these features:

- Task-Oriented: CogDL focuses on tasks on graphs and provides corresponding models, datasets, and leaderboards.
- Easy-Running: CogDL supports running multiple experiments simultaneously on multiple models and datasets under a specific task using multiple GPUs.
- Multiple Tasks: CogDL supports node classification and link prediction tasks on homogeneous/heterogeneous networks, as well as graph classification.
- Extensibility: You can easily add new datasets, models and tasks and conduct experiments for them!

- Supported tasks:

   - Node classification
   - Link prediction
   - Graph classification
   - Community detection (testing)
   - Social influence prediction (testing)
   - Graph reasoning (todo)
   - Graph pre-training (todo)
   - Combinatorial optimization on graphs (todo)

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
