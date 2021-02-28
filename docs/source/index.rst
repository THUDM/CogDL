.. CogDL documentation master file

Welcome to CogDL's Documentation!
=================================

.. image:: _static/cogdl-logo.png

CogDL is a graph representation learning toolkit that allows researchers and developers to easily train and compare baseline or customized models for node classification, graph classification, and other important tasks in the graph domain. 

We summarize the contributions of CogDL as follows:

  - **High Efficiency**: CogDL utilizes well-optimized operators to speed up training and save GPU memory of GNN models.
  - **Easy-to-Use**: CogDL provides easy-to-use APIs for running experiments with the given models and datasets using hyper-parameter search.
  - **Extensibility**: The design of CogDL makes it easy to apply GNN models to new scenarios based on our framework.
  - **Reproducibility**: CogDL provides reproducible leaderboards for state-of-the-art models on most of important tasks in the graph domain.

Supported tasks:

   - Node classification
   - Link prediction
   - Graph classification
   - Graph pre-training
   - Graph clustering
   - Graph similarity search

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   install
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Advanced Guides

   task/index
   trainer
   model
   dataset

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api/data
   api/datasets
   api/tasks
   api/models
   api/layers
   api/options
   api/utils
   api/experiments
   api/pipelines


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`