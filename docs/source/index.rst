.. CogDL documentation master file

Welcome to CogDL's Documentation!
=================================

.. image:: _static/cogdl-logo.png

CogDL is a graph representation learning toolkit that allows researchers and developers to easily train and compare baseline or customized models for node classification, graph classification, and other important tasks in the graph domain. 

We summarize the contributions of CogDL as follows:

- **Efficiency**: CogDL utilizes well-optimized operators to speed up training and save GPU memory of GNN models.
- **Ease of Use**: CogDL provides easy-to-use APIs for running experiments with the given models and datasets using hyper-parameter search.
- **Extensibility**: The design of CogDL makes it easy to apply GNN models to new scenarios based on our framework.

❗ News
------------

- The new **v0.5.3 release** supports mixed-precision training by setting \textit{fp16=True} and provides a basic [example](https://github.com/THUDM/cogdl/blob/master/examples/jittor/gcn.py) written by [Jittor](https://github.com/Jittor/jittor). It also updates the tutorial in the document, fixes downloading links of some datasets, and fixes potential bugs of operators. 
- The new **v0.5.2 release** adds a GNN example for ogbn-products and updates geom datasets. It also fixes some potential bugs including setting devices, using cpu for inference, etc.
- The new **v0.5.1 release** adds fast operators including SpMM (cpu version) and scatter_max (cuda version). It also adds lots of datasets for node classification. 🎉
- The new **v0.5.0 release** designs and implements a unified training loop for GNN. It introduces `DataWrapper` to help prepare the training/validation/test data and `ModelWrapper` to define the training/validation/test steps. 
- The new **v0.4.1 release** adds the implementation of Deep GNNs and the recommendation task. It also supports new pipelines for generating embeddings and recommendation. Welcome to join our tutorial on KDD 2021 at 10:30 am - 12:00 am, Aug. 14th (Singapore Time). More details can be found in https://kdd2021graph.github.io/. 🎉
- The new **v0.4.0 release** refactors the data storage (from ``Data`` to ``Graph``) and provides more fast operators to speed up GNN training. It also includes many self-supervised learning methods on graphs. BTW, we are glad to announce that we will give a tutorial on KDD 2021 in August. Please see this `link <https://kdd2021graph.github.io/>`_ for more details. 🎉
- The new **v0.3.0 release** provides a fast spmm operator to speed up GNN training. We also release the first version of `CogDL paper <https://arxiv.org/abs/2103.00959>`_ in arXiv. You can join `our slack <https://join.slack.com/t/cogdl/shared_invite/zt-b9b4a49j-2aMB035qZKxvjV4vqf0hEg>`_ for discussion. 🎉🎉🎉
- The new **v0.2.0 release** includes easy-to-use ``experiment`` and ``pipeline`` APIs for all experiments and applications. The ``experiment`` API supports automl features of searching hyper-parameters. This release also provides ``OAGBert`` API for model inference (``OAGBert`` is trained on large-scale academic corpus by our lab). Some features and models are added by the open source community (thanks to all the contributors 🎉).
- The new **v0.1.2 release** includes a pre-training task, many examples, OGB datasets, some knowledge graph embedding methods, and some graph neural network models. The coverage of CogDL is increased to 80%. Some new APIs, such as ``Trainer`` and ``Sampler``, are developed and being tested. 
- The new **v0.1.1 release** includes the knowledge link prediction task, many state-of-the-art models, and ``optuna`` support. We also have a `Chinese WeChat post <https://mp.weixin.qq.com/s/IUh-ctQwtSXGvdTij5eDDg>`_ about the CogDL release.

Citing CogDL
------------

Please cite `our paper <https://arxiv.org/abs/2103.00959>`_ if you find our code or results useful for your research:

::

   @article{cen2021cogdl,
      title={CogDL: A Toolkit for Deep Learning on Graphs},
      author={Yukuo Cen and Zhenyu Hou and Yan Wang and Qibin Chen and Yizhen Luo and Zhongming Yu and Hengrui Zhang and Xingcheng Yao and Aohan Zeng and Shiguang Guo and Yuxiao Dong and Yang Yang and Peng Zhang and Guohao Dai and Yu Wang and Chang Zhou and Hongxia Yang and Jie Tang},
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
   :caption: Tutorials 

   tutorial/graph
   tutorial/training
   tutorial/custom_dataset
   tutorial/custom_gnn
   tutorial/results

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api/data
   api/datasets
   api/models
   api/data_wrappers
   api/model_wrappers
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