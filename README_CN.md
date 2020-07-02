![CogDL](docs/source/_static/cogdl-logo.png)
===

[![Build Status](https://travis-ci.org/THUDM/cogdl.svg?branch=master)](https://travis-ci.org/THUDM/cogdl)
[![Coverage Status](https://coveralls.io/repos/github/THUDM/cogdl/badge.svg?branch=master)](https://coveralls.io/github/THUDM/cogdl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/cogdl/badge/?version=latest)](https://cogdl.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/thudm/cogdl)](https://github.com/THUDM/cogdl/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**[主页](http://keg.cs.tsinghua.edu.cn/cogdl/cn)** | **[智源链接](http://open.baai.ac.cn/cogdl-toolkit)** | **[文档](https://cogdl.readthedocs.io)** | **[海报](https://qibinc.github.io/cogdl-leaderboard/poster_cn.pdf)** | **[English](./README.md)**

CogDL是由清华大学计算机系知识工程实验室（KEG）开发的基于图的深度学习的研究工具，基于Python语言和[Pytorch](https://github.com/pytorch/pytorch)库。CogDL允许研究人员和开发人员可以轻松地训练和比较基线算法或自定义模型，以进行结点分类，链接预测，图分类，社区发现等基于图结构的任务。 它提供了许多流行模型的实现，包括：非图神经网络算法例如Deepwalk、LINE、Node2vec、NetMF、ProNE、methpath2vec、PTE、graph2vec、DGK等；图神经网络算法例如GCN、GAT、GraphSAGE、FastGCN、GTN、HAN、GIN、DiffPool等。它也提供了一些下游任务，包括结点分类（分为是否具有节点属性），链接预测（分为同构和异构），图分类（分有监督和⽆监督）以及为这些任务构建各种算法效果的排行榜。

CogDL的特性包括：


- 任务导向： CogDL以图上的任务为主，提供了相关的模型、数据集以及我们得到的排行榜。
- 一键运行： CogDL支持用户使用多个GPU同时运行同一个任务下多个模型在多个数据集上的多组实验。
- 多类任务： CogDL支持同构/异构网络中的节点分类和链接预测任务以及图分类任务。
- 可扩展性：用户可以基于CogDL已有的框架来实现和提交新的数据集、模型和任务。


## CogDL的整体框架


![avatar](cogdl_cn.png)

CogDL的整体框架如上图所示，针对不同的任务，CogDL支持以下模型：

*   无监督结点分类: ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512), DNGR [(Cao et al, AAAI'16)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715).  
    
*   半监督结点分类: Graph U-Net [(Gao et al., 2019)](https://arxiv.org/abs/1905.05178), MixHop [(Abu-El-Haija et al., ICML'19)](https://arxiv.org/abs/1905.00067), DR-GAT [(Zou et al., 2019)](https://arxiv.org/abs/1907.02237), GAT [(Veličković et al., ICLR'18)](https://arxiv.org/abs/1710.10903), DGI [(Veličković et al., ICLR'19)](https://arxiv.org/abs/1809.10341), GCN [(Kipf et al., ICLR'17)](https://arxiv.org/abs/1609.02907), GraphSAGE [(Hamilton et al., NeurIPS'17)](https://arxiv.org/abs/1706.02216), Chebyshev [(Defferrard et al., NeurIPS'16)](https://arxiv.org/abs/1606.09375).  
    
*   异构结点分类: GTN [(Yun et al, NeurIPS'19)](https://arxiv.org/abs/1911.06455), HAN [(Xiao et al, WWW'19)](https://arxiv.org/abs/1903.07293), PTE [(Tang et al, KDD'15)](https://arxiv.org/abs/1508.00200), Metapath2vec [(Dong et al, KDD'17)](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf), Hin2vec [(Fu et al, CIKM'17)](https://dl.acm.org/doi/10.1145/3132847.3132953).  
    
*   链接预测: ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156), SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf).  
    
*   多重边链接预测: GATNE [(Cen et al, KDD'19)](https://arxiv.org/abs/1905.01669), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512).  
    
*   无监督图分类: Infograph [(Sun et al, ICLR'20)](https://openreview.net/forum?id=r1lfF2NYvH), Graph2Vec [(Narayanan et al, CoRR'17)](https://arxiv.org/abs/1707.05005), DGK [(Yanardag et al, KDD'15)](https://dl.acm.org/doi/10.1145/2783258.2783417).  
    
*   有监督图分类: GIN [(Xu et al, ICLR'19)](https://openreview.net/forum?id=ryGs6iA5Km), DiffPool [(Ying et al, NeuIPS'18)](https://arxiv.org/abs/1806.08804), SortPool [(Zhang et al, AAAI'18)](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf), PATCH\_SAN [(Niepert et al, ICML'16)](https://arxiv.org/pdf/1605.05273.pdf), DGCNN [(Wang et al, ACM Transactions on Graphics'17)](https://arxiv.org/abs/1801.07829).

## 模型

CogDL实现了一系列不同类型的模型，下面列出了这些算法的特性。

### 无监督结点表示学习的算法

| Algorithm |      Directed      |       Weight       |  Shallow network   | Matrix factorization |      Sampling      |  Reproducibility   |    GPU support     |
| --------- | :----------------: | :----------------: | :----------------: | :------------------: | :----------------: | :----------------: | :----------------: |
| DeepWalk  |                    |                    | :heavy_check_mark: |                      |                    | :heavy_check_mark: |                    |
| LINE      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                      | :heavy_check_mark: | :heavy_check_mark: |                    |
| Node2vec  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                      | :heavy_check_mark: | :heavy_check_mark: |                    |
| SDNE      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                      |                    | :heavy_check_mark: | :heavy_check_mark: |
| DNGR      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                      |                    |                    | :heavy_check_mark: |
| HOPE      | :heavy_check_mark: | :heavy_check_mark: |                    |  :heavy_check_mark:  |                    | :heavy_check_mark: |                    |
| GraRep    | :heavy_check_mark: | :heavy_check_mark: |                    |  :heavy_check_mark:  |                    |                    |                    |
| NetMF     | :heavy_check_mark: | :heavy_check_mark: |                    |  :heavy_check_mark:  |                    | :heavy_check_mark: |                    |
| NetSMF    |                    | :heavy_check_mark: |                    |  :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |                    |
| ProNE     | :heavy_check_mark: | :heavy_check_mark: |                    |  :heavy_check_mark:  |                    | :heavy_check_mark: |                    |


其中，在Reproducibility项为空的算法，表示setting不一致或暂时没有完全复现。

### 半监督结点表示学习的算法

| Algorithm   |       Weight       |      Sampling      |     Attention      |     Inductive      |  Reproducibility   |    GPU support     |
| ----------- | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
| Graph U-Net | :heavy_check_mark: | :heavy_check_mark: |                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| MixHop      | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| Dr-GAT      |                    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| GAT         |                    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DGI         | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| GCN         | :heavy_check_mark: |                    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| GraphSAGE   | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Chebyshev   | :heavy_check_mark: |                    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### 异构结点表示学习的算法

| Algorithm    |     Multi-Node     |     Multi-Edge     |     Attribute      |     Supervised     |      MetaPath      |  Reproducibility   |    GPU support     |
| ------------ | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
| GATNE        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Metapath2vec | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: |                    |
| PTE          | :heavy_check_mark: |                    |                    |                    |                    | :heavy_check_mark: |                    |
| Hin2vec      | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| GTN          | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| HAN          | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 图表示学习的算法

| Algorithm  |    Node feature    |    Unsupervised    |    Graph kernel    |  Shallow network   |  Reproducibility   |    GPU support     |
| ---------- | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
| Infograph  | :heavy_check_mark: | :heavy_check_mark: |                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| Diffpool   | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| Graph2Vec  |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| Sortpool   | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| GIN        | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| PATCHY_SAN | :heavy_check_mark: |                    | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |
| DGCNN      | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| DGK        |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |                    |

## 排行榜

CogDL提供了一些下游任务，包括结点分类（具有或不具有结点属性），链接预测（具有或不具有属性，异构或非异构）和图分类（有监督或无监督）任务。 我们建立了几个排行榜，这些排行榜列出了各类算法在这些任务上的最新结果。

### 结点分类



#### 无监督多标签结点分类

这是一个根据无监督的多标签结点分类设置而构建的排行榜，我们在几个真实的数据集上运行CogDL上的无监督表示学习算法，并将输出的表示和90％的结点标签作为经L2归一化的逻辑回归中的训练数据，使用剩余10％的标签作为测试数据，计算并按照Micro-F1的大小进行排序。

| Rank | Method                                                       |    PPI    | Blogcatalog | Wikipedia |
| ---- | ------------------------------------------------------------ | :-------: | :---------: | :-------: |
| 1    | ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf) | **26.32** |  **43.63**  |   57.64   |
| 2    | NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971) |   24.86   |    43.49    | **58.46** |
| 3    | Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754) |   23.86   |    42.51    |   53.68   |
| 4    | NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156) |   24.39   |    43.21    |   51.42   |
| 5    | DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652) |   22.72   |    42.26    |   50.42   |
| 6    | LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578) |   23.15   |    39.29    |   49.83   |
| 7    | Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751) |   23.24   |    35.52    |   52.96   |
| 8    | SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf) |   20.14   |    40.32    |   48.24   |
| 9    | GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512) |   20.96   |    34.35    |   51.84   |
| 10   | DNGR [(Cao et al, AAAI'16)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715) |   16.45   |    28.54    |   48.57   |

#### 半监督有属性的结点分类

下面是几种常见的图神经网络算法在半监督结点分类任务上构建的排行榜。我们在经典的三个数据集Cora, Citeseer和Pubmed进行了实验，以Accuracy指标来评价模型的效果。

| Rank | Method                                                       |      Cora      |    Citeseer    |     Pubmed     |
| ---- | ------------------------------------------------------------ | :------------: | :------------: | :------------: |
| 1    | Graph U-Net [(Gao et al., 2019)](https://arxiv.org/abs/1905.05178) | **84.4 ± 0.6** | **73.2 ± 0.5** |   79.6 ± 0.2   |
| 2    | MixHop [(Abu-El-Haija et al., ICML'19)](https://arxiv.org/abs/1905.00067) |   81.9 ± 0.4   |   71.4 ± 0.8   | **80.8 ± 0.6** |
| 3    | DR-GAT [(Zou et al., 2019)](https://arxiv.org/abs/1907.02237) |   83.6 ± 0.5   |   72.8 ± 0.8   |   79.1 ± 0.3   |
| 4    | GAT [(Veličković et al., ICLR'18)](https://arxiv.org/abs/1710.10903) |   83.0 ± 0.7   |   72.5 ± 0.7   |   79.0 ± 0.3   |
| 5    | DGI [(Veličković et al., ICLR'19)](https://arxiv.org/abs/1809.10341) |   82.3 ± 0.6   |   71.8 ± 0.7   |   76.8 ± 0.6   |
| 6    | GCN [(Kipf et al., ICLR'17)](https://arxiv.org/abs/1609.02907) |   81.4 ± 0.5   |   70.9 ± 0.5   |   79.0 ± 0.3   |
| 7    | GraphSAGE [(Hamilton et al., NeurIPS'17)](https://arxiv.org/abs/1706.02216) |   80.1 ± 0.2   |   66.2 ± 0.4   |   76.9 ± 0.7   |
| 8    | Chebyshev [(Defferrard et al., NeurIPS'16)](https://arxiv.org/abs/1606.09375) |   79.2 ± 1.4   |   69.3 ± 1.3   |   68.5 ± 1.2   |

#### 异构结点分类

对于异构的结点分类任务，我们使用Macro F1来评价模型的效果。我们在GTN算法的实验设置和数据集下对所有算法进行评估。

| Rank | Method                                                       |   DBLP    |    ACM    |   IMDB    |
| ---- | ------------------------------------------------------------ | :-------: | :-------: | :-------: |
| 1    | GTN [(Yun et al, NeurIPS'19)](https://arxiv.org/abs/1911.06455) | **92.03** | **90.85** | **59.24** |
| 2    | HAN [(Xiao et al, WWW'19)](https://arxiv.org/abs/1903.07293) |   91.21   |   87.25   |   53.94   |
| 3    | PTE [(Tang et al, KDD'15)](https://arxiv.org/abs/1508.00200) |   78.65   |   87.44   |   48.91   |
| 4    | Metapath2vec [(Dong et al, KDD'17)](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) |   75.18   |   88.79   |   43.10   |
| 5    | Hin2vec [(Fu et al, CIKM'17)](https://dl.acm.org/doi/10.1145/3132847.3132953) |   74.31   |   84.66   |   44.04   |



### 链接预测

#### 链接预测

对于链接预测任务，我们通过隐去数据集中10%的边，然后对隐去的边进行预测，使用ROC-AUC指标来评估模型的性能。ROC-AUC指标代表了一条随机未观察到的边对应的两个结点比一条随机不存在的边对应的两个结点更相似的概率。

| Rank | Method                                                       |    PPI    |   Wikipedia   |
| ---- | ------------------------------------------------------------ | :-------: | :-----------: |
| 1    | ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf) |   79.93   |   **82.74**   |
| 2    | NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971) |   79.04   |     73.24     |
| 3    | Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751) | **80.21** |     68.89     |
| 4    | LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578) |   73.75   |     66.51     |
| 5    | Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754) |   70.19   |     66.60     |
| 6    | NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156) |   68.64   |     67.52     |
| 7    | DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652) |   69.65   |     65.93     |
| 8    | SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf) |   54.87   |     60.72     |



#### 异构链接预测

对于异构链接预测任务，我们会对数据集中的某些视图下的链接进行预测，然后取Macro ROC-AUC作为评价指标。我们提出的GATNE模型是专门针对这种多视图的异构网络，而这里列举的其他方法只能处理同构网络，因此我们向这些方法分别输入不同视图下的网络，并为每种视图下的网络分别获得结点表示用于链接预测，最后同样采用Macro ROC-AUC作为评测指标。

| Rank | Method                                                       |  Amazon   |  YouTube  |  Twitter  |
| ---- | ------------------------------------------------------------ | :-------: | :-------: | :-------: |
| 1    | GATNE [(Cen et al, KDD'19)](https://arxiv.org/abs/1905.01669) |   97.44   | **84.61** | **92.30** |
| 2    | NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971) | **97.72** |   82.53   |   73.75   |
| 3    | ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf) |   96.51   |   78.96   |   81.32   |
| 4    | Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754) |   86.86   |   74.01   |   78.30   |
| 5    | DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652) |   92.54   |   74.31   |   60.29   |
| 6    | LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578) |   92.56   |   73.40   |   60.36   |
| 7    | Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751) |   94.39   |   74.66   |   70.61   |
| 8    | GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512) |   83.88   |   71.37   |   49.64   |



### 图分类

CogDL统一对有监督和无监督的图分类算法在相同的若干个真实的数据集上运行和评测。有监督图分类算法使用kfold对算法进行调参、训练和评测；无监督图分类算法学习到图的表示之后，将其作为输入并利用90%的图的标签作为SVM的训练数据，使用剩余10%的标签作为测试数据。两者均计算并按照Accuracy的大小进行排序。

| Rank | Method                                                       |   MUTAG   |   IMDB-B   |   IMDB-M   |   PROTEINS   |   COLLAB   |
| :--- | :----------------------------------------------------------- | :-------: | :--------: | :--------: | :----------: | :--------: |
| 1    | Infograph [(Sun et al, ICLR'20)](https://openreview.net/forum?id=r1lfF2NYvH) | **88.95** | 74.50  | 51.33  |  73.93   | 78.14  |
| 2    | GIN [(Xu et al, ICLR'19)](https://openreview.net/forum?id=ryGs6iA5Km) | 88.33 | **76.70**  | 50.80  |  72.86   | 79.52  |
| 3    | DiffPool [(Ying et al, NeuIPS'18)](https://arxiv.org/abs/1806.08804) | 85.18 | 74.30  | 50.73  |  75.30   | 77.20  |
| 4    | SortPool [(Zhang et al, AAAI'18)](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) | 85.61 | 75.20  | 51.07  |  74.11   | 79.98  |
| 5    | Graph2Vec [(Narayanan et al, CoRR'17)](https://arxiv.org/abs/1707.05005) | 83.68 | 73.90  | **52.27**  |  73.30   | **85.58**  |
| 6    | PATCH_SAN [(Niepert et al, ICML'16)](https://arxiv.org/pdf/1605.05273.pdf) | 85.12 | 76.00  | 46.20  |  **75.50**   | 75.42  |
| 7    | DGCNN [(Wang et al, ACM Transactions on Graphics'17)](https://arxiv.org/abs/1801.07829) | 83.33 | 69.50  | 46.33  |  66.67   | 77.45  |
| 8    | DGK [(Yanardag et al, KDD'15)](https://dl.acm.org/doi/10.1145/2783258.2783417) | 83.68 | 55.00  | 40.40  |  72.59   |   /    |


## 使用说明：

CogDL安装请按照这里的说明来安装PyTorch和其他依赖项:

- https://github.com/pytorch/pytorch#installation
- https://github.com/rusty1s/pytorch_geometric/#installation
- pip install -e .

基本用法可以使用 `python train.py --task example_task --dataset example_dataset --model example_method` 来在 `example_data` 上运行 `example_method` 并使用 `example_task` 来评测结果。

- --task, 运行的任务名称，像node_classification, unsupervised_node_classification, link_prediction这样来评测表示质量的下游任务。
- --dataset, 运行的数据集名称，可以是以空格分隔开的数据集名称的列表,现在支持的数据集包括 cora, citeseer, pumbed, PPI, wikipedia, blogcatalog, dblp, flickr等。
- --model, 运行的模型名称,可以是个列表，支持的模型包括 gcn, gat,deepwalk, node2vec, hope, grarep, netmf, netsmf, prone等。

如果你想在Cora数据集上运行GCN模型,并用node classification评测,可以使用如下指令:
`python train.py --task node_classification --dataset cora --model gcn`


## 自定义数据集或模型

- 提交你的先进算法：如果您有一个性能优异的算法并愿意发布出来，你可以在我们的代码仓库里提出一个[issue](https://github.com/qibinc/cognitive_graph/issues)。在验证该算法的原创性，创造性和效果后，我们将该算法的效果添加到我们的排行榜上。
- 添加你自己的数据集：如果您有一个独特，有研究价值的数据集并且愿意发布出来，你可以在我们的代码仓库里提出一个[issue](https://github.com/qibinc/cognitive_graph/issues)，我们将把所以适合的模型在您的数据集上运行并更新我们的排行榜。
- 实现你自己的模型：如果您有一个性能优秀的算法，并愿意在我们的工具包中实现它，以帮助更多的人，您可以创建一个pull request，详细信息可见[该页面](https://help.github.com/en/articles/creating-a-pull-request)。

如果您在我们的工具包或自定义步骤中遇到任何困难，请随时提出一个github issue或发表评论。您可以在24小时内得到答复。



## 参考文献

[1] Sun, Fan-Yun, Jordan Hoffmann, and Jian Tang. "InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization." arXiv preprint arXiv:1908.01000 (2019).

[2] Qiu, Jiezhong, Yuxiao Dong, Hao Ma, Jian Li, Chi Wang, Kuansan Wang, and Jie Tang. "Netsmf: Large-scale network embedding as sparse matrix factorization." In The World Wide Web Conference, pp. 1509-1520. 2019.

[3] Cen, Yukuo, Xu Zou, Jianwei Zhang, Hongxia Yang, Jingren Zhou, and Jie Tang. "Representation learning for attributed multiplex heterogeneous network." In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1358-1368. 2019.

[4] Zhang, Jie, Yuxiao Dong, Yan Wang, Jie Tang, and Ming Ding. "ProNE: fast and scalable network representation learning." In Proc. 28th Int. Joint Conf. Artif. Intell., IJCAI, pp. 4278-4284. 2019.

[5] Zou, Xu, Qiuye Jia, Jianwei Zhang, Chang Zhou, Hongxia Yang, and Jie Tang. "Dimensional Reweighting Graph Convolutional Networks." arXiv preprint arXiv:1907.02237 (2019).

[6] Gao, Hongyang, and Shuiwang Ji. "Graph u-nets." arXiv preprint arXiv:1905.05178 (2019).

[7] Abu-El-Haija, Sami, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, and Aram Galstyan. "Mixhop: Higher-order graph convolutional architectures via sparsified neighborhood mixing." arXiv preprint arXiv:1905.00067 (2019).

[8] Veličković, Petar, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, and R. Devon Hjelm. "Deep graph infomax." arXiv preprint arXiv:1809.10341 (2018).

[9] Yun, Seongjun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J. Kim. "Graph Transformer Networks." In Advances in Neural Information Processing Systems, pp. 11960-11970. 2019.

[10] Wang, Xiao, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S. Yu. "Heterogeneous graph attention network." In The World Wide Web Conference, pp. 2022-2032. 2019.

[11] Xu, Keyulu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).

[12] Qiu, Jiezhong, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, and Jie Tang. "Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 459-467. 2018.

[13] Veličković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).

[14] Ying, Zhitao, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. "Hierarchical graph representation learning with differentiable pooling." In Advances in neural information processing systems, pp. 4800-4810. 2018.

[15] Zhang, Muhan, Zhicheng Cui, Marion Neumann, and Yixin Chen. "An end-to-end deep learning architecture for graph classification." In Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

[16] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).

[17] Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." In Advances in neural information processing systems, pp. 1024-1034. 2017.

[18] Dong, Yuxiao, Nitesh V. Chawla, and Ananthram Swami. "metapath2vec: Scalable representation learning for heterogeneous networks." In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 135-144. 2017.

[19] Fu, Tao-yang, Wang-Chien Lee, and Zhen Lei. "Hin2vec: Explore meta-paths in heterogeneous information networks for representation learning." In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, pp. 1797-1806. 2017.

[20] Narayanan, Annamalai, Mahinthan Chandramohan, Rajasekar Venkatesan, Lihui Chen, Yang Liu, and Shantanu Jaiswal. "graph2vec: Learning distributed representations of graphs." arXiv preprint arXiv:1707.05005 (2017).

[21] Wang, Yue, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. "Dynamic graph cnn for learning on point clouds." ACM Transactions on Graphics (TOG) 38, no. 5 (2019): 1-12.

[22] Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 855-864. 2016.

[23] Ou, Mingdong, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. "Asymmetric transitivity preserving graph embedding." In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 1105-1114. 2016.

[24] Wang, Daixin, Peng Cui, and Wenwu Zhu. "Structural deep network embedding." In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 1225-1234. 2016.

[25] Cao, Shaosheng, Wei Lu, and Qiongkai Xu. "Deep neural networks for learning graph representations." In Thirtieth AAAI conference on artificial intelligence. 2016.

[26] Defferrard, Michaël, Xavier Bresson, and Pierre Vandergheynst. "Convolutional neural networks on graphs with fast localized spectral filtering." In Advances in neural information processing systems, pp. 3844-3852. 2016.

[27] Niepert, Mathias, Mohamed Ahmed, and Konstantin Kutzkov. "Learning convolutional neural networks for graphs." In International conference on machine learning, pp. 2014-2023. 2016.

[28] Tang, Jian, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. "Line: Large-scale information network embedding." In Proceedings of the 24th international conference on world wide web, pp. 1067-1077. 2015.

[29] Cao, Shaosheng, Wei Lu, and Qiongkai Xu. "Grarep: Learning graph representations with global structural information." In Proceedings of the 24th ACM international on conference on information and knowledge management, pp. 891-900. 2015.

[30] Tang, Jian, Meng Qu, and Qiaozhu Mei. "Pte: Predictive text embedding through large-scale heterogeneous text networks." In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1165-1174. 2015.

[31] Yanardag, Pinar, and S. V. N. Vishwanathan. "Deep graph kernels." In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1365-1374. 2015.

[32] Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 701-710. 2014.
