Models of cogdl
======================


Introduction to graph representation learning
---------------------------------------------
Inspired by recent trends of representation learning on computer vision and natural language processing, graph representation learning is proposed as an efficient technique to address this issue. Graph representation aims at either learning low-dimensional continuous vectors for vertices/graphs while preserving intrinsic graph properties, or using graph encoders to an end-to-end training.

Recently, graph neural networks (GNNs) have been proposed and have achieved impressive performance in semi-supervised representation learning. Graph Convolution Networks (GCNs) proposes a convolutional architecture via a localized first-order approximation of spectral graph convolutions. GraphSAGE is a general inductive framework that leverages node features to generate node embeddings for previously unseen samples. Graph Attention Networks (GATs) utilizes the multi-head self-attention mechanism and enables (implicitly) specifying different weights to different nodes in a neighborhood.

CogDL now supports the following tasks:
-----------------------
- unsupervised node classification
- semi-supervised node classification
- heterogeneous node classification
- link prediction
- multiplex link prediction
- unsupervised graph classification
- supervised graph classification
- graph pre-training
- attributed graph clustering

Unsupervised Multi-label Node Classification
---------------------------------------------

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
---------------------------------------------------


===================================================================================== ==================
Model                                                                                  Name in Cogdl
===================================================================================== ==================
Grand(`Feng et al.,NLPS'20 <https://arxiv.org/pdf/2005.11079.pdf>`__)                     grand
GCNII((`Chen et al.,ICML’20  <https://arxiv.org/pdf/2007.02133.pdf>`__)                   gcnii
DR-GAT `(Zou et al., 2019) <https://arxiv.org/abs/1907.02237>`__                          drgat
MVGRL `(Hassani et al., KDD’20) <https://arxiv.org/pdf/2006.05582v1.pdf>`__               mvgrl
APPNP `(Klicpera et al., ICLR’19) <https://arxiv.org/pdf/2006.05582v1.pdf>`__             ppnp
Graph U-Net `(Gao et al., 2019) <https://arxiv.org/abs/1905.05178>`__
GAT `(Veličković et al., ICLR’18) <https://arxiv.org/abs/1710.10903>`__                   gat
GDC_GCN `(Klicpera et al., NeurIPS’19) <https://arxiv.org/pdf/1911.05485.pdf>`__          gdc_gcn
DropEdge `(Rong et al., ICLR’20) <https://openreview.net/pdf?id=Hkx1qkrKPr>`__             dropedge_gcn
GCN `(Kipf et al., ICLR’17) <https://arxiv.org/abs/1609.02907>`__                         gcn
DGI `(Veličković et al., ICLR’19) <https://arxiv.org/abs/1809.10341>`__                   dgi
JK-net `(Xu et al., ICML’18) <https://arxiv.org/pdf/1806.03536.pdf>`__
GraphSAGE `(Hamilton et al., NeurIPS’17) <https://arxiv.org/abs/1706.02216>`__            graphsage
GraphSAGE `(unsup)(Hamilton et al., NeurIPS’17) <https://arxiv.org/abs/1706.02216>`__      unsup_graphsage
Chebyshev `(Defferrard et al., NeurIPS’16) <https://arxiv.org/abs/1606.09375>`__
MixHop  `(Abu-El-Haija et al., ICML’19) <https://arxiv.org/abs/1905.00067>`__             mixhop
===================================================================================== ==================

Multiplex Node Classification
-----------------------------

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
_________________________

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
RotatE `(Sun et al, ICLR’19) <https://arxiv.org/pdf/1902.10197.pdf>`__
ComplEx `(Trouillon et al, ICML’18) <https://arxiv.org/abs/1606.06357>`__
TransE `(Bordes et al, NIPS’13)Bordes et al, NIPS'13)] <https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>`__
DistMult `(Yang et al, ICLR’15) <https://arxiv.org/pdf/1412.6575.pdf>`__
CompGCN `(Vashishth et al, ICLR’20) <https://arxiv.org/abs/1911.03082>`__                                                                                      compgcn
======================================================================================================================================================== ==================


Graph Classification
____________________

==================================================================================================== ===============
 Model                                                                                                Name in Cogdl
==================================================================================================== ===============
GIN `(Xu et al, ICLR’19) <https://openreview.net/forum?id=ryGs6iA5Km>`__                                 gin
Infograph `(Sun et al, ICLR’20) <https://openreview.net/forum?id=r1lfF2NYvH>`__                         infograph
DiffPool `(Ying et al, NeuIPS’18) <https://arxiv.org/abs/1806.08804>`__                                  diffpool
SortPool `(Zhang et al, AAAI’18) <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`__        softpool
Graph2Vec `(Narayanan et al, CoRR’17) <https://arxiv.org/abs/1707.05005>`__                              graph2vec
PATCH_SAN `(Niepert et al, ICML’16) <https://arxiv.org/pdf/1605.05273.pdf>`__                            patchy_san
HGP-SL `(Zhang et al, AAAI’20) <https://arxiv.org/abs/1911.05954>`__
DGCNN `(Wang et al, ACM Transactions on Graphics’17) <https://arxiv.org/abs/1801.07829>`__
SAGPool `(J. Lee, ICML’19) <https://arxiv.org/abs/1904.08082>`__
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

