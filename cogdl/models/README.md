Models of CogDL
===============

CogDL now supports the following models for different tasks:

- unsupervised node classification (无监督结点分类): ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512), DNGR [(Cao et al, AAAI'16)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715).

- semi-supervised node classification (半监督结点分类): SGC-PN [(Zhao & Akoglu, 2019)](https://arxiv.org/abs/1909.12223), Graph U-Net [(Gao et al., 2019)](https://arxiv.org/abs/1905.05178), MixHop [(Abu-El-Haija et al., ICML'19)](https://arxiv.org/abs/1905.00067), DR-GAT [(Zou et al., 2019)](https://arxiv.org/abs/1907.02237), GAT [(Veličković et al., ICLR'18)](https://arxiv.org/abs/1710.10903), DGI [(Veličković et al., ICLR'19)](https://arxiv.org/abs/1809.10341), GCN [(Kipf et al., ICLR'17)](https://arxiv.org/abs/1609.02907), GraphSAGE [(Hamilton et al., NeurIPS'17)](https://arxiv.org/abs/1706.02216), Chebyshev [(Defferrard et al., NeurIPS'16)](https://arxiv.org/abs/1606.09375).

- heterogeneous node classification (异构结点分类): GTN [(Yun et al, NeurIPS'19)](https://arxiv.org/abs/1911.06455), HAN [(Xiao et al, WWW'19)](https://arxiv.org/abs/1903.07293), PTE [(Tang et al, KDD'15)](https://arxiv.org/abs/1508.00200), Metapath2vec [(Dong et al, KDD'17)](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf), Hin2vec [(Fu et al, CIKM'17)](https://dl.acm.org/doi/10.1145/3132847.3132953).

- multiplex link prediction (多重边链接预测): GATNE [(Cen et al, KDD'19)](https://arxiv.org/abs/1905.01669), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512).

- unsupervised graph classification (无监督图分类): Infograph [(Sun et al, ICLR'20)](https://openreview.net/forum?id=r1lfF2NYvH), Graph2Vec [(Narayanan et al, CoRR'17)](https://arxiv.org/abs/1707.05005), DGK [(Yanardag et al, KDD'15)](https://dl.acm.org/doi/10.1145/2783258.2783417).

- supervised graph classification (有监督图分类): GIN [(Xu et al, ICLR'19)](https://openreview.net/forum?id=ryGs6iA5Km), DiffPool [(Ying et al, NeuIPS'18)](https://arxiv.org/abs/1806.08804), SortPool [(Zhang et al, AAAI'18)](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf), PATCHY_SAN [(Niepert et al, ICML'16)](https://arxiv.org/pdf/1605.05273.pdf), DGCNN [(Wang et al, ACM Transactions on Graphics'17)](https://arxiv.org/abs/1801.07829).

> `metis` is required to run ClusterGCN, you can follow the following steps to install `metis`.
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
6) pip install metis
