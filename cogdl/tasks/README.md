Tasks and Leaderboards
======================

CogDL now supports the following tasks:
- unsupervised node classification
- semi-supervised node classification
- heterogeneous node classification
- link prediction
- multiplex link prediction
- unsupervised graph classification
- supervised graph classification
- graph pre-training
- attributed graph clustering
- graph similarity search

## Leaderboard

CogDL provides several downstream tasks including node classification (with or without node attributes), link prediction (with or without attributes, heterogeneous or not). These leaderboards maintain state-of-the-art results and benchmarks on these tasks.

All models have been implemented in [models](https://github.com/THUDM/cogdl/tree/master/cogdl/models) and the hyperparameters to reproduce the following results have been put in [examples](https://github.com/THUDM/cogdl/tree/master/examples). 


### Node Classification

#### Unsupervised Multi-label Node Classification

This leaderboard reports unsupervised multi-label node classification setting. we run all algorithms on several real-world datasets and report the sorted experimental results (Micro-F1 score with 90% labels as training data in L2 normalization logistic regression).

| Rank | Method                                                                                                           |     PPI      |  Wikipedia   | Blogcatalog  |     DBLP     |    Flickr    |
| ---- | ---------------------------------------------------------------------------------------------------------------- | :----------: | :----------: | :----------: | :----------: | :----------: |
| 1    | NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971)                                                    | 23.73 ± 0.22 | 57.42 ± 0.56 | 42.47 ± 0.35 | 56.72 ± 0.14 | 36.27 ± 0.17 |
| 2    | ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf)                                 | 24.60 ± 0.39 | 56.06 ± 0.48 | 41.14 ± 0.26 | 54.05 ± 0.44 | 36.56 ± 0.11 |
| 3    | NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156)                                                   | 21.67 ± 0.34 | 53.81 ± 0.58 | 34.76 ± 0.33 | 58.62 ± 0.25 | 35.49 ± 0.07 |
| 4    | Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754)                           | 20.67 ± 0.54 | 54.59 ± 0.51 | 40.16 ± 0.29 | 57.36 ± 0.39 | 36.13 ± 0.13 |
| 5    | LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578)                                                     | 21.82 ± 0.56 | 52.46 ± 0.26 | 38.06 ± 0.39 | 49.78 ± 0.37 | 31.61 ± 0.09 |
| 6    | DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652)                                               | 20.74 ± 0.40 | 49.53 ± 0.54 | 40.48 ± 0.47 | 57.54 ± 0.32 | 36.09 ± 0.10 |
| 7    | Spectral [(Tang et al, Data Min Knowl Disc (2011))](https://link.springer.com/article/10.1007/s10618-010-0210-x) | 22.48 ± 0.30 | 49.35 ± 0.34 | 41.41 ± 0.34 | 43.68 ± 0.58 | 33.09 ± 0.07 |
| 8    | Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751)                                   | 21.43 ± 0.32 | 54.04 ± 0.47 | 33.99 ± 0.35 | 56.15 ± 0.22 | 28.97 ± 0.19 |
| 9    | GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512)                               | 20.60 ± 0.34 | 54.37 ± 0.40 | 33.48 ± 0.30 | 52.76 ± 0.42 | 31.83 ± 0.12 |

#### Semi-Supervised Node Classification with Attributes

This leaderboard reports the semi-supervised node classification under a transductive setting including several popular graph neural network methods.

| Rank | Method                                                                            |      Cora      |    Citeseer    |     Pubmed     |
| ---- | --------------------------------------------------------------------------------- | :------------: | :------------: | :------------: |
| 1    | Grand([Feng et al., NIPS'20](https://arxiv.org/pdf/2005.11079.pdf))               |   84.8 ± 0.3   | **75.1 ± 0.3** | **82.4 ± 0.4** |
| 2    | GCNII([Chen et al., ICML'20](https://arxiv.org/pdf/2007.02133.pdf))               | **85.1 ± 0.3** |   71.3 ± 0.4   |   80.2 ± 0.3   |
| 3    | DR-GAT [(Zou et al., 2019)](https://arxiv.org/abs/1907.02237)                     |   83.6 ± 0.5   |   72.8 ± 0.8   |   79.1 ± 0.3   |
| 4    | MVGRL [(Hassani et al., KDD'20)](https://arxiv.org/pdf/2006.05582v1.pdf)          |   83.6 ± 0.2   |   73.0 ± 0.3   |   80.1 ± 0.7   |
| 5    | APPNP [(Klicpera et al., ICLR'19)](https://arxiv.org/pdf/1810.05997.pdf)          |   82.5 ± 0.8   |   71.2 ± 0.2   |   80.2 ± 0.2   |
| 6    | GAT [(Veličković et al., ICLR'18)](https://arxiv.org/abs/1710.10903)              |   82.9 ± 0.8   |   71.0 ± 0.3   |   78.9 ± 0.3   |
| 7    | GCN [(Kipf et al., ICLR'17)](https://arxiv.org/abs/1609.02907)                    |   82.3 ± 0.3   |   71.4 ± 0.4   |   79.5 ± 0.2   |
| 8    | SRGCN                                                                             |   82.2 ± 0.2   |   72.8 ± 0.2   |   79.0 ± 0.4   |
| 9    | DGI [(Veličković et al., ICLR'19)](https://arxiv.org/abs/1809.10341)              |   82.0 ± 0.2   |   71.2 ± 0.4   |   76.5 ± 0.6   |
| 10   | GraphSAGE [(Hamilton et al., NeurIPS'17)](https://arxiv.org/abs/1706.02216)       |   80.1 ± 0.2   |   66.2 ± 0.4   |   77.2 ± 0.7   |
| 11   | GraphSAGE(unsup)[(Hamilton et al., NeurIPS'17)](https://arxiv.org/abs/1706.02216) |   78.2 ± 0.9   |   65.8 ± 1.0   |   78.2 ± 0.7   |
| 12   | Chebyshev [(Defferrard et al., NeurIPS'16)](https://arxiv.org/abs/1606.09375)     |   79.0 ± 1.0   |   69.8 ± 0.5   |   68.6 ± 1.0   |
| 13   | Graph U-Net [(Gao et al., 2019)](https://arxiv.org/abs/1905.05178)                |      81.8      |      67.1      |      77.3      |
| 14   | MixHop [(Abu-El-Haija et al., ICML'19)](https://arxiv.org/abs/1905.00067)         |   81.9 ± 0.4   |   71.4 ± 0.8   |   80.8 ± 0.6   |
| 15   | SGC-PN [(Zhao & Akoglu, 2019)](https://arxiv.org/abs/1909.12223)                  |   76.4 ± 0.3   |   64.6 ± 0.6   |   79.6 ± 0.3   |

#### Multiplex Node Classification

For multiplex node classification, we use macro F1 to evaluate models. We evaluate all models under the setting and datasets of GTN.

| Rank | Method                                                                                                             |   DBLP    |    ACM    |   IMDB    |
| ---- | ------------------------------------------------------------------------------------------------------------------ | :-------: | :-------: | :-------: |
| 1    | GTN [(Yun et al, NeurIPS'19)](https://arxiv.org/abs/1911.06455)                                                    | **92.03** | **90.85** | **59.24** |
| 2    | HAN [(Xiao et al, WWW'19)](https://arxiv.org/abs/1903.07293)                                                       |   91.21   |   87.25   |   53.94   |
| 3    | GCC [(Qiu et al, KDD'20)](http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD20-Qiu-et-al-GCC-GNN-pretrain.pdf) |   79.42   |   86.82   |   55.86   |
| 4    | PTE [(Tang et al, KDD'15)](https://arxiv.org/abs/1508.00200)                                                       |   78.65   |   87.44   |   48.91   |
| 5    | Metapath2vec [(Dong et al, KDD'17)](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)  |   75.18   |   88.79   |   43.10   |
| 6    | Hin2vec [(Fu et al, CIKM'17)](https://dl.acm.org/doi/10.1145/3132847.3132953)                                      |   74.31   |   84.66   |   44.04   |

### Link Prediction

#### Link Prediction

For link prediction, we adopt Area Under the Receiver Operating Characteristic Curve (ROC AUC), which represents the probability that vertices in a random unobserved link are more similar than those in a random nonexistent link. We evaluate these measures while removing 10 percents of edges on these dataset. We repeat our experiments for 10 times and report the results in order.

| Rank | Method                                                                                     |    PPI    | Wikipedia |
| ---- | ------------------------------------------------------------------------------------------ | :-------: | :-------: |
| 1    | ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf)           |   79.93   | **82.74** |
| 2    | NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971)                              |   79.04   |   73.24   |
| 3    | Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751)             | **80.21** |   68.89   |
| 4    | LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578)                               |   73.75   |   66.51   |
| 5    | Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754)     |   70.19   |   66.60   |
| 6    | NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156)                             |   68.64   |   67.52   |
| 7    | DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652)                         |   69.65   |   65.93   |
| 8    | SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf) |   54.87   |   60.72   |

#### Multiplex Link Prediction

For multiplex link prediction, we adopt Area Under the Receiver Operating Characteristic Curve (ROC AUC). We evaluate these measures while removing 15 percents of edges on these dataset. We repeat our experiments for 10 times and report the three matrices in order.

| Rank | Method                                                                                 |  Amazon   |  YouTube  |  Twitter  |
| ---- | -------------------------------------------------------------------------------------- | :-------: | :-------: | :-------: |
| 1    | GATNE [(Cen et al, KDD'19)](https://arxiv.org/abs/1905.01669)                          |   97.44   | **84.61** | **92.30** |
| 2    | NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971)                          | **97.72** |   82.53   |   73.75   |
| 3    | ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf)       |   96.51   |   78.96   |   81.32   |
| 4    | Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754) |   86.86   |   74.01   |   78.30   |
| 5    | DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652)                     |   92.54   |   74.31   |   60.29   |
| 6    | LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578)                           |   92.56   |   73.40   |   60.36   |
| 7    | Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751)         |   94.39   |   74.66   |   70.61   |
| 8    | GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512)     |   83.88   |   71.37   |   49.64   |

#### Knowledge graph completion

For knowledge graph completion, we adopt Mean Reciprocal Rank (MRR) as the evaluation metric. Every triplet-based embedding algorithm is trained with 80000 steps, negative sample size 128 and learning rate 0.001. Every GNN-based embedding algorithm is trained with 3000 steps with patience 20.

| Rank | Method                                                                                                                      | FB15k-237 |   WN18    |  WN18RR   |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | :-------: | :-------: | :-------: |
| 1    | RotatE [(Sun et al, ICLR'19)](https://arxiv.org/pdf/1902.10197.pdf)                                                         | **31.10** | **93.99** | **46.05** |
| 2    | ComplEx [(Trouillon et al, ICML'18)](https://arxiv.org/abs/1606.06357)                                                      |   15.38   |   86.40   |   40.77   |
| 3    | TransE [(Bordes et al, NIPS'13)](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) |   30.50   |   71.55   |   21.85   |
| 4    | DistMult [(Yang et al, ICLR'15)](https://arxiv.org/pdf/1412.6575.pdf)                                                       |   15.91   |   75.60   |   39.06   |
| 5    | CompGCN [(Vashishth et al, ICLR'20)](https://arxiv.org/abs/1911.03082)                                                      |   12.66   |   36.34   |   27.20   |

### Graph Classification

This leaderboard reports the performance of graph classification methods. we run all algorithms on several datasets and report the sorted experimental results.

| Rank | Method                                                                                         |   MUTAG   |  IMDB-B   |  IMDB-M   | PROTEINS  |  COLLAB   |
| :--- | :--------------------------------------------------------------------------------------------- | :-------: | :-------: | :-------: | :-------: | :-------: |
| 1    | GIN [(Xu et al, ICLR'19)](https://openreview.net/forum?id=ryGs6iA5Km)                          | **92.06** | **76.10** |   51.80   |   75.19   |   79.52   |
| 2    | Infograph [(Sun et al, ICLR'20)](https://openreview.net/forum?id=r1lfF2NYvH)                   |   88.95   |   74.50   |   51.33   |   73.93   |   79.4    |
| 3    | DiffPool [(Ying et al, NeuIPS'18)](https://arxiv.org/abs/1806.08804)                           |   85.18   |   72.40   |   50.50   |   75.30   |   79.27   |
| 4    | SortPool [(Zhang et al, AAAI'18)](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf) |   87.25   |   75.40   |   50.47   |   73.23   |   80.07   |
| 5    | Graph2Vec [(Narayanan et al, CoRR'17)](https://arxiv.org/abs/1707.05005)                       |   83.68   |   73.90   | **52.27** |   73.30   | **85.58** |
| 6    | PATCH_SAN [(Niepert et al, ICML'16)](https://arxiv.org/pdf/1605.05273.pdf)                     |   86.12   |   76.00   |   46.40   | **75.38** |   74.34   |
| 7    | HGP-SL [(Zhang et al, AAAI'20)](https://arxiv.org/abs/1911.05954)                              |   81.93   |   74.00   |   49.53   |   73.94   |   82.08   |
| 8    | DGCNN [(Wang et al, ACM Transactions on Graphics'17)](https://arxiv.org/abs/1801.07829)        |   83.33   |   69.50   |   46.33   |   66.67   |   77.45   |
| 9    | SAGPool [(J. Lee, ICML'19)](https://arxiv.org/abs/1904.08082)                                  |   55.55   |   63.00   |   51.33   |   72.59   |     /     |
| 10   | DGK [(Yanardag et al, KDD'15)](https://dl.acm.org/doi/10.1145/2783258.2783417)                 |   83.68   |   55.00   |   40.40   |   72.59   |     /     |
