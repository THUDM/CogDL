Model Training
==============

Introduction to graph representation learning
---------------------------------------------

Inspired by recent trends of representation learning on computer vision and natural language processing, graph representation learning is proposed as an efficient technique to address this issue. 
Graph representation aims at either learning low-dimensional continuous vectors for vertices/graphs while preserving intrinsic graph properties, or using graph encoders to an end-to-end training.

Recently, graph neural networks (GNNs) have been proposed and have achieved impressive performance in semi-supervised representation learning. 
Graph Convolution Networks (GCNs) proposes a convolutional architecture via a localized first-order approximation of spectral graph convolutions. 
GraphSAGE is a general inductive framework that leverages node features to generate node embeddings for previously unseen samples. 
Graph Attention Networks (GATs) utilizes the multi-head self-attention mechanism and enables (implicitly) specifying different weights to different nodes in a neighborhood.

Models of CogDL
---------------

CogDL now supports the following models for different tasks:

-  unsupervised node classification (无监督结点分类): ProNE `(Zhang et
   al, IJCAI’19) <https://www.ijcai.org/Proceedings/2019/0594.pdf>`__,
   NetMF `(Qiu et al, WSDM’18) <http://arxiv.org/abs/1710.02971>`__,
   Node2vec `(Grover et al,
   KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939754>`__,
   NetSMF `(Qiu et at, WWW’19) <https://arxiv.org/abs/1906.11156>`__,
   DeepWalk `(Perozzi et al,
   KDD’14) <http://arxiv.org/abs/1403.6652>`__, LINE `(Tang et al,
   WWW’15) <http://arxiv.org/abs/1503.03578>`__, Hope `(Ou et al,
   KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939751>`__,
   SDNE `(Wang et al,
   KDD’16) <https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf>`__,
   GraRep `(Cao et al,
   CIKM’15) <http://dl.acm.org/citation.cfm?doid=2806416.2806512>`__,
   DNGR `(Cao et al,
   AAAI’16) <https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715>`__.

-  semi-supervised node classification (半监督结点分类): SGC-PN `(Zhao &
   Akoglu, 2019) <https://arxiv.org/abs/1909.12223>`__, Graph U-Net
   `(Gao et al., 2019) <https://arxiv.org/abs/1905.05178>`__, MixHop
   `(Abu-El-Haija et al.,
   ICML’19) <https://arxiv.org/abs/1905.00067>`__, DR-GAT `(Zou et al.,
   2019) <https://arxiv.org/abs/1907.02237>`__, GAT `(Veličković et al.,
   ICLR’18) <https://arxiv.org/abs/1710.10903>`__, DGI `(Veličković et
   al., ICLR’19) <https://arxiv.org/abs/1809.10341>`__, GCN `(Kipf et
   al., ICLR’17) <https://arxiv.org/abs/1609.02907>`__, GraphSAGE
   `(Hamilton et al., NeurIPS’17) <https://arxiv.org/abs/1706.02216>`__,
   Chebyshev `(Defferrard et al.,
   NeurIPS’16) <https://arxiv.org/abs/1606.09375>`__.

-  heterogeneous node classification (异构结点分类): GTN `(Yun et al,
   NeurIPS’19) <https://arxiv.org/abs/1911.06455>`__, HAN `(Xiao et al,
   WWW’19) <https://arxiv.org/abs/1903.07293>`__, PTE `(Tang et al,
   KDD’15) <https://arxiv.org/abs/1508.00200>`__, Metapath2vec `(Dong et
   al,
   KDD’17) <https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf>`__,
   Hin2vec `(Fu et al,
   CIKM’17) <https://dl.acm.org/doi/10.1145/3132847.3132953>`__.

-  multiplex link prediction (多重边链接预测): GATNE `(Cen et al,
   KDD’19) <https://arxiv.org/abs/1905.01669>`__, NetMF `(Qiu et al,
   WSDM’18) <http://arxiv.org/abs/1710.02971>`__, ProNE `(Zhang et al,
   IJCAI’19) <https://www.ijcai.org/Proceedings/2019/0594.pdf>`__,
   Node2vec `(Grover et al,
   KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939754>`__,
   DeepWalk `(Perozzi et al,
   KDD’14) <http://arxiv.org/abs/1403.6652>`__, LINE `(Tang et al,
   WWW’15) <http://arxiv.org/abs/1503.03578>`__, Hope `(Ou et al,
   KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939751>`__,
   GraRep `(Cao et al,
   CIKM’15) <http://dl.acm.org/citation.cfm?doid=2806416.2806512>`__.

-  unsupervised graph classification (无监督图分类): Infograph `(Sun et
   al, ICLR’20) <https://openreview.net/forum?id=r1lfF2NYvH>`__,
   Graph2Vec `(Narayanan et al,
   CoRR’17) <https://arxiv.org/abs/1707.05005>`__, DGK `(Yanardag et al,
   KDD’15) <https://dl.acm.org/doi/10.1145/2783258.2783417>`__.

-  supervised graph classification (有监督图分类): GIN `(Xu et al,
   ICLR’19) <https://openreview.net/forum?id=ryGs6iA5Km>`__, DiffPool
   `(Ying et al, NeuIPS’18) <https://arxiv.org/abs/1806.08804>`__,
   SortPool `(Zhang et al,
   AAAI’18) <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`__,
   PATCHY_SAN `(Niepert et al,
   ICML’16) <https://arxiv.org/pdf/1605.05273.pdf>`__, DGCNN `(Wang et
   al, ACM Transactions on
   Graphics’17) <https://arxiv.org/abs/1801.07829>`__.

..

   ``metis`` is required to run ClusterGCN, you can follow the following
   steps to install ``metis``. 1) Download metis-5.1.0.tar.gz from
   http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
   2) cd metis-5.1.0 3) make config shared=1 prefix=~/.local/ 4) make
   install 5) export METIS_DLL=~/.local/lib/libmetis.so 6) pip install
   metis

Unified Trainer
---------------
CogDL provides a unified trainer for GNN models, which takes over the entire loop of the training process. The unified trainer, which contains much engineering code, is implemented flexibly to cover arbitrary GNN training settings. 

.. image:: ../_static/cogdl-training.png

We design four decoupled modules for the GNN training, including *Model*, *Model Wrapper*, *Dataset*, *Data Wrapper*. The *Model Wrapper* is for the training and testing steps, while the *Data Wrapper* is designed to construct data loaders used by *Model Wrapper*. 

 
The main contributions of most GNN papers mainly lie on three modules except *Dataset*, as shown in the table. 
For example, the GCN paper trains the GCN model under the (semi-)supervised and full-graph setting, while the DGI paper trains the GCN model by maximizing local-global mutual information. 
The training method of the DGI is considered as a model wrapper named *dgi\_mw*, which could be used for other scenarios. 

============== ======== ================ ====================
Paper          Model    Model Wrapper    Data Wrapper       
============== ======== ================ ====================
GCN            GCN      supervised       full-graph          
GAT            GAT      supervised       full-graph          
GraphSAGE      SAGE     sage\_mw         neighbor sampling   
Cluster-GCN    GCN      supervised       graph clustering    
DGI            GCN      dgi\_mw          full-graph          
============== ======== ================ ====================


Based on the design of the unified trainer and decoupled modules, we could do arbitrary combinations of models, model wrappers, and data wrappers. For example, if we want to apply DGI to large-scale datasets, all we need is to substitute the full-graph data wrapper with the neighbor-sampling or clustering data wrappers without additional modifications. 
If we propose a new GNN model, all we need is to write essential PyTorch-style code for the model. The rest could be automatically handled by CogDL by specifying the model wrapper and the data wrapper. 
We could quickly conduct experiments for the model using the trainer via \textit{trainer = Trainer(epochs,...)} and \textit{trainer.run(...)}. 
Moreover, based on the unified trainer, CogDL provides native support for many useful features, including hyperparameter optimization, efficient training techniques, and experiment management without any modification to the model implementation. 



Experiment API
--------------
CogDL provides a more easy-to-use API upon *Trainer*, i.e., *experiment*. 
We take node classification as an example and show how to use CogDL to finish a workflow using GNN. In supervised setting, node classification aims to predict the ground truth label for each node. 
CogDL provides abundant of common benchmark datasets and GNN models. On the one hand, you can simply start a running using
models and datasets in CogDL. This is convenient when you want to test the reproducibility of proposed GNN or get baseline
results in different datasets.

.. code-block:: python

    from cogdl import experiment
    experiment(model="gcn", dataset="cora")



Or you can create each component separately and manually run the process using ``build_dataset``, ``build_model`` in CogDL.

.. code-block:: python

    from cogdl import experiment
    from cogdl.datasets import build_dataset
    from cogdl.models import build_model
    from cogdl.options import get_default_args 

    args = get_default_args(model="gcn", dataset="cora")
    dataset = build_dataset(args)
    model = build_model(args)
    experiment(model=model, dataset=dataset)


As show above, model/dataset are key components in establishing a training process. In fact, CogDL also supports
customized model and datasets. This will be introduced in next chapter. In the following we will briefly show the details
of each component.


How to save trained model?
--------------------------

CogDL supports saving the trained model with ``checkpoint_path`` in command line or API usage. For example:

.. code-block:: python

    experiment(model="gcn", dataset="cora", checkpoint_path="gcn_cora.pt")


When the training stops, the model will be saved in `gcn_cora.pt`. If you want to continue the training from previous checkpoint
with different parameters(such as learning rate, weight decay and etc.), keep the same model parameters (such as hidden size, model layers)
and do it as follows:


.. code-block:: python

    experiment(model="gcn", dataset="cora", checkpoint_path="gcn_cora.pt", resume_training=True)


In command line usage, the same results can be achieved with ``--checkpoint-path {path}`` and ``--resume-training``.


How to save embeddings?
-----------------------
Graph representation learning (network embedding and unsupervised GNNs) aims to get node representation. The embeddings
can be used in various downstream applications. CogDL will save node embeddings in the given path specified by ``--save-emb-path {path}``. 

.. code-block:: python

    experiment(model="prone", dataset="blogcatalog", save_emb_path="./embeddings/prone_blog.npy")


Evaluation on node classification will run as the end of training. We follow the same experimental settings used in DeepWalk, Node2Vec and ProNE.
We randomly sample different percentages of labeled nodes for training a liblinear classifier and use the remaining for testing
We repeat the training for several times and report the average Micro-F1. By default, CogDL samples 90% labeled nodes for training
for one time. You are expected to change the setting with ``--num-shuffle`` and ``--training-percents`` to your needs.

In addition, CogDL supports evaluating node embeddings without training in different evaluation settings. The following
code snippet evaluates the embedding we get above:

.. code-block:: python

    experiment(
        model="prone",
        dataset="blogcatalog",
        load_emb_path="./embeddings/prone_blog.npy",
        num_shuffle=5,
        training_percents=[0.1, 0.5, 0.9]
    )



You can also use command line to achieve the same results

.. code-block:: bash

    # Get embedding
    python script/train.py --model prone --dataset blogcatalog

    # Evaluate only
    python script/train.py --model prone --dataset blogcatalog --load-emb-path ./embeddings/prone_blog.npy --num-shuffle 5 --training-percents 0.1 0.5 0.9

