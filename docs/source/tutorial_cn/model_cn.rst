Cogdl中的模型
=============

图表示学习介绍
-------------
受最近计算机视觉和自然语言处理方面的表示学习趋势的启发，图表示学习被提出。图表示旨在学习顶点/图的低维连续向量，同时保留内在
图属性，或者使用图编码器进行端到端训练。
最近，已经提出了图神经网络（GNN），并在半监督表示学习中取得了令人印象深刻的性能。图卷积网络 (GCN) 通过谱图卷积的局部一阶近似提出了一种卷积架构。Gra
phSAGE 是一个通用归纳框架，它利用节点特征为以前未见过的样本生成节点embeddings。Graph Attention Networks (GATs) 利用多头自注意力机制，并能够（隐式）
为邻域中的不同节点指定不同的权重。

CogDL现在支持以下任务
-------------------
- unsupervised node classification(无监督节点分类)
- semi-supervised node classification(半监督节点分类)
- heterogeneous node classification(异构节点分类)
- link prediction(链接预测)
- multiplex link prediction(多路链接预测)
- unsupervised graph classification(无监督图分类)
- supervised graph classification(监督图分类)
- graph pre-training(图预训练)
- attributed graph clustering(属性图聚类)

CogDL 提供了丰富的通用基准数据集和 GNN 模型。您可以使用 CogDL 中的模型和数据集简单地开始运行。

.. code-block:: python

    from cogdl import experiment
    experiment(model="gcn", dataset="cora")

Unsupervised Multi-label Node Classification
____________________________________________

==================================================================================================================== ================
                                       Model                                                                          Name in Cogdl
==================================================================================================================== ================
NetMF `(Qiu et al, WSDM’18) <http://arxiv.org/abs/1710.02971>`__                                                          netmf
ProNE `(Zhang et al, IJCAI’19) <https://www.ijcai.org/Proceedings/2019/0594.pdf>`__                                      prone
NetSMF `(Qiu et at, WWW’19) <https://arxiv.org/abs/1906.11156>`__                                                         netsmf
Node2vec `(Grover et al, KDD’16) <http://dl.acm .org/citation.cfm?doid=2939672.2939754>`__                               node2vec
LINE `(Tang et al, WWW’15) <http://arxiv.org/abs/1503.03578>`__                                                          line
DeepWalk `(Perozzi et al, KDD’14) <http://arxiv.org/abs/1403.6652>`__                                                    deepwalk
Spectral `(Tang et al, Data Min Knowl Disc (2011)) <https://link.springer.com/article/10.1007/s10618-010-0210-x>`__      spectral
Hope `(Ou et al, KDD’16) <http://dl.acm .org/citation.cfm?doid=2939672.2939751>`__                                       hope
GraRep `(Cao et al, CIKM’15) <http://dl.acm.org/citation.cfm?doid=2806416.2806512>`__                                    grarep
==================================================================================================================== ================

Semi-Supervised Node Classification with Attributes
___________________________________________________


===================================================================================== ==================
Model                                                                                  Name in Cogdl
===================================================================================== ==================
Grand(`Feng et al.,NLPS'20 <https://arxiv.org/pdf/2005.11079.pdf>`__)                     grand
GCNII((`Chen et al.,ICML’20  <https://arxiv.org/pdf/2007.02133.pdf>`__)                   gcnii
DR-GAT `(Zou et al., 2019) <https://arxiv.org/abs/1907.02237>`__                          drgat
MVGRL `(Hassani et al., KDD’20) <https://arxiv.org/pdf/2006.05582v1.pdf>`__               mvgrl
APPNP `(Klicpera et al., ICLR’19) <https://arxiv.org/pdf/2006.05582v1.pdf>`__             ppnp
GAT `(Veličković et al., ICLR’18) <https://arxiv.org/abs/1710.10903>`__                   gat
GDC_GCN `(Klicpera et al., NeurIPS’19) <https://arxiv.org/pdf/1911.05485.pdf>`__          gdc_gcn
DropEdge `(Rong et al., ICLR’20) <https://openreview.net/pdf?id=Hkx1qkrKPr>`__            dropedge_gcn
GCN `(Kipf et al., ICLR’17) <https://arxiv.org/abs/1609.02907>`__                         gcn
DGI `(Veličković et al., ICLR’19) <https://arxiv.org/abs/1809.10341>`__                   dgi
GraphSAGE `(Hamilton et al., NeurIPS’17) <https://arxiv.org/abs/1706.02216>`__            graphsage
GraphSAGE `(unsup)(Hamilton et al., NeurIPS’17) <https://arxiv.org/abs/1706.02216>`__     unsup_graphsage
MixHop  `(Abu-El-Haija et al., ICML’19) <https://arxiv.org/abs/1905.00067>`__             mixhop
===================================================================================== ==================

Multiplex Node Classification
______________________________

======================================================================================================================= =================
         Model                                                                                                           Name in Cogdl
======================================================================================================================= =================
Simple-HGN `(Lv and Ding et al, KDD’21) <https://github.com/THUDM/HGB>`__                                                  `simple-hgn <https://github.com/QingFei1/cogdl/tree/master/examples/simple_hgn>`__
GTN `(Yun et al, NeurIPS’19) <https://arxiv.org/abs/1911.06455>`__                                                         gtn
HAN `(Xiao et al, WWW’19) <https://arxiv.org/abs/1903.07293>`__                                                            han
GCC `(Qiu et al, KDD’20) <http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD20-Qiu-et-al-GCC-GNN-pretrain.pdf>`__      gcc
PTE `(Tang et al, KDD’15) <https://arxiv.org/abs/1508.00200>`__                                                            pte
Metapath2vec `(Dong et al, KDD’17) <https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf>`__       metapath2vec
Hin2vec `(Fu et al, CIKM’17) <https://dl.acm.org/doi/10.1145/3132847.3132953>`__                                           hin2vec
======================================================================================================================= =================


Link Prediction
_______________

================================================================================================ =============
 Model                                                                                           Name in Cogdl
================================================================================================ =============
ProNE `(Zhang et al, IJCAI’19) <https://www.ijcai.org/Proceedings/2019/0594.pdf>`__                 prone
NetMF `(Qiu et al, WSDM’18) <http://arxiv.org/abs/1710.02971>`__                                    netmf
Hope `(Ou et al, KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939751>`__                   hope
LINE `(Tang et al, WWW’15) <http://arxiv.org/abs/1503.03578>`__                                     line
Node2vec `(Grover et al, KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939754>`__           node2vec
NetSMF `(Qiu et at, WWW’19) <https://arxiv.org/abs/1906.11156>`__                                   netsmf
DeepWalk `(Perozzi et al, KDD’14) <http://arxiv.org/abs/1403.6652>`__                               deepwalk
SDNE `(Wang et al, KDD’16) <https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf>`__       sdne

================================================================================================ =============


Multiplex Link Prediction
___________________________

============================================================================================ =============
 Model                                                                                       Name in Cogdl
============================================================================================ =============
GATNE `(Cen et al, KDD’19) <https://arxiv.org/abs/1905.01669>`__                                gatne
NetMF `(Qiu et al, WSDM’18) <http://arxiv.org/abs/1710.02971>`__                                netmf
ProNE `(Zhang et al, IJCAI’19) <https://www.ijcai.org/Proceedings/2019/0594.pdf>`__             prone++
Node2vec `(Grover et al, KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939754>`__       node2vec
DeepWalk `(Perozzi et al, KDD’14) <http://arxiv.org/abs/1403.6652>`__                           deepwalk
LINE `(Tang et al, WWW’15) <http://arxiv.org/abs/1503.03578>`__                                 line
Hope `(Ou et al, KDD’16) <http://dl.acm.org/citation.cfm?doid=2939672.2939751>`__               hope
GraRep `(Cao et al, CIKM’15) <http://dl.acm.org/citation.cfm?doid=2806416.2806512>`__           grarep
============================================================================================ =============

Knowledge graph completion
__________________________


======================================================================================================================================================== ==================
 Model                                                                                                                                                     Name in Cogdl
======================================================================================================================================================== ==================
CompGCN `(Vashishth et al, ICLR’20) <https://arxiv.org/abs/1911.03082>`__                                                                                      compgcn
======================================================================================================================================================== ==================


Graph Classification
_______________________

==================================================================================================== ===============
 Model                                                                                                Name in Cogdl
==================================================================================================== ===============
GIN `(Xu et al, ICLR’19) <https://openreview.net/forum?id=ryGs6iA5Km>`__                                 gin
Infograph `(Sun et al, ICLR’20) <https://openreview.net/forum?id=r1lfF2NYvH>`__                         infograph
DiffPool `(Ying et al, NeuIPS’18) <https://arxiv.org/abs/1806.08804>`__                                  diffpool
SortPool `(Zhang et al, AAAI’18) <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`__        softpool
Graph2Vec `(Narayanan et al, CoRR’17) <https://arxiv.org/abs/1707.05005>`__                              graph2vec
PATCH_SAN `(Niepert et al, ICML’16) <https://arxiv.org/pdf/1605.05273.pdf>`__                            patchy_san
DGK `(Yanardag et al, KDD’15) <https://dl.acm.org/doi/10.1145/2783258.2783417>`__                        dgk

==================================================================================================== ===============



Attributed graph clustering
___________________________

==================================================================================================== ===============
 Model                                                                                                Name in Cogdl
==================================================================================================== ===============
AGC `(Zhang et al, IJCAI 19) <https://arxiv.org/abs/1906.01210>`__                                       agc
DAEGC `(Wang et al, ICLR’20) <https://arxiv.org/pdf/1906.06532.pdf>`__                                   daegc
==================================================================================================== ===============

