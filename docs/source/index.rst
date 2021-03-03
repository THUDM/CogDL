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

❗ News
------------

- The new **v0.3.0 release** provides a fast spmm operator to speed up GNN training. We also release the first version of `CogDL paper <https://arxiv.org/abs/2103.00959>`_ in arXiv. You can join `our slack <https://join.slack.com/t/cogdl/shared_invite/zt-b9b4a49j-2aMB035qZKxvjV4vqf0hEg>`_ for discussion. 🎉🎉🎉
- The new **v0.2.0 release** includes easy-to-use ``experiment`` and ``pipeline`` APIs for all experiments and applications. The ``experiment`` API supports automl features of searching hyper-parameters. This release also provides ``OAGBert`` API for model inference (``OAGBert`` is trained on large-scale academic corpus by our lab). Some features and models are added by the open source community (thanks to all the contributors 🎉).
- The new **v0.1.2 release** includes a pre-training task, many examples, OGB datasets, some knowledge graph embedding methods, and some graph neural network models. The coverage of CogDL is increased to 80%. Some new APIs, such as ``Trainer`` and ``Sampler``, are developed and being tested. 
- The new **v0.1.1 release** includes the knowledge link prediction task, many state-of-the-art models, and ``optuna`` support. We also have a `Chinese WeChat post <https://mp.weixin.qq.com/s/IUh-ctQwtSXGvdTij5eDDg>`_ about the CogDL release.

Citing CogDL
------------

Please cite `our paper <https://arxiv.org/abs/2103.00959>`_ if you find our code or results useful for your research:

::

   @article{cen2021cogdl,
      title={CogDL: An Extensive Toolkit for Deep Learning on Graphs},
      author={Yukuo Cen and Zhenyu Hou and Yan Wang and Qibin Chen and Yizhen Luo and Xingcheng Yao and Aohan Zeng and Shiguang Guo and Peng Zhang and Guohao Dai and Yu Wang and Chang Zhou and Hongxia Yang and Jie Tang},
      journal={arXiv preprint arXiv:2103.00959},
      year={2021}
   }


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