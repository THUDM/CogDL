![CogDL](http://keg.cs.tsinghua.edu.cn/cogdl/cogdl-logo.png)
===

[![Build Status](https://travis-ci.org/THUDM/cogdl.svg?branch=master)](https://travis-ci.org/THUDM/cogdl)
[![Coverage Status](https://coveralls.io/repos/github/THUDM/cogdl/badge.svg?branch=master)](https://coveralls.io/github/THUDM/cogdl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/cogdl/badge/?version=latest)](https://cogdl.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/thudm/cogdl)](https://github.com/THUDM/cogdl/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**[Homepage](http://keg.cs.tsinghua.edu.cn/cogdl)** | **[BAAI link](http://open.baai.ac.cn/cogdl-toolkit)** | **[Documentation](https://cogdl.readthedocs.io)** | **[Poster](https://qibinc.github.io/cogdl-leaderboard/poster.pdf)** | **[中文](./README_CN.md)**

CogDL is a graph representation learning toolkit that allows researchers and developers to easily train and compare baseline or custom models for node classification, link prediction and other tasks on graphs. It provides implementations of many popular models, including: non-GNN Baselines like Deepwalk, LINE, NetMF, GNN Baselines like GCN, GAT, GraphSAGE.

Note that CogDL is **still actively under development**, so feedback and contributions are welcome.
Feel free to submit your contributions as a pull request.

CogDL features:

- Task-Oriented: CogDL focuses on tasks on graphs and provides corresponding models, datasets, and leaderboards.
- Easy-Running: CogDL supports running multiple experiments simultaneously on multiple models and datasets under a specific task using multiple GPUs.
- Multiple Tasks: CogDL supports node classification and link prediction tasks on homogeneous/heterogeneous networks, as well as graph classification.
- Extensibility: You can easily add new datasets, models and tasks and conduct experiments for them!
- Supported tasks:
  - Node classification
  - Link prediction
  - Graph classification
  - Graph reasoning (todo)
  - Graph pre-training (todo)
  - Combinatorial optimization on graphs (todo)

## Getting Started

## Requirements and Installation

- PyTorch version >= 1.0.0
- Python version >= 3.6
- PyTorch Geometric
- Deep Graph Library

Please follow the instructions here to install PyTorch: https://github.com/pytorch/pytorch#installation, PyTorch Geometric https://github.com/rusty1s/pytorch_geometric/#installation and Deep Graph Library https://docs.dgl.ai/install/index.html.

Install other dependencies:

```bash
pip install -e .
```

## Usage

You can use `python scripts/train.py --task example_task --dataset example_dataset --model example_method` to run example_method on example_data and evaluate it via example_task.

### General parameters

- --task, downstream tasks to evaluate representation like node_classification, unsupervised_node_classification, link_prediction. More tasks can be found in the [cogdl/tasks](https://github.com/THUDM/cogdl/tree/master/cogdl/tasks).
- --dataset, dataset name to run, can be a list of datasets with space like `cora citeseer ppi`. Supported datasets include
'cora', 'citeseer', 'pumbed', 'PPI', 'wikipedia', 'blogcatalog', 'flickr'. More datasets can be found in the [cogdl/datasets](https://github.com/THUDM/cogdl/tree/master/cogdl/datasets).
- --model, model name to run, can be a list of models like `deepwalk line prone`. Supported datasets include
'gcn', 'gat', 'graphsage', 'deepwalk', 'node2vec', 'hope', 'grarep', 'netmf', 'netsmf', 'prone'. More models can be found in the [cogdl/models](https://github.com/THUDM/cogdl/tree/master/cogdl/models).

For example, if you want to run Deepwalk, Line, Netmf on Wikipedia with node classification task, with 5 different seeds:

```bash
$ python scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model line netmf --seed 0 1 2 3 4
```

Expected output:

| Variant                | Micro-F1 0.1   | Micro-F1 0.3   | Micro-F1 0.5   | Micro-F1 0.7   | Micro-F1 0.9   |
|------------------------|----------------|----------------|----------------|----------------|----------------|
| ('wikipedia', 'line')  | 0.4069±0.0011  | 0.4071±0.0010  | 0.4055±0.0013  | 0.4054±0.0020  | 0.4080±0.0042  |
| ('wikipedia', 'netmf') | 0.4551±0.0024  | 0.4932±0.0022  | 0.5046±0.0017  | 0.5084±0.0057  | 0.5125±0.0035  |

If you want to run parallel experiments on your server with multiple GPUs on multiple models gcn, gat on multiple datasets Cora, Citeseer with node classification task:

```bash
$ python scripts/parallel_train.py --task node_classification --dataset cora --model pyg_gcn pyg_gat --device-id 0 1 --seed 0 1 2 3 4
```

Expected output:

| Variant             | Acc           |
|---------------------|---------------|
| ('cora', 'pyg_gcn') | 0.7922±0.0082 |
| ('cora', 'pyg_gat') | 0.8092±0.0055 |


## Model Characteristics

We summarize the characteristics of all methods for different tasks in the following, where reproducibility means whether the model is reproduced in our experimental setting currently. 

### Unsupervised Graph Embedding Methods

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


### Graph Neural Networks

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

### Heterogeneous Graph Embedding Methods

| Algorithm    |     Multi-Node     |     Multi-Edge     |     Attribute      |     Supervised     |      MetaPath      |  Reproducibility   |    GPU support     |
| ------------ | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
| GATNE        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Metapath2vec | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: |                    |
| PTE          | :heavy_check_mark: |                    |                    |                    |                    | :heavy_check_mark: |                    |
| Hin2vec      | :heavy_check_mark: |                    |                    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| GTN          | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| HAN          | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### Methods for Graph Classification

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


## Leaderboard

CogDL provides several downstream tasks including node classification (with or without node attributes), link prediction (with or without attributes, heterogeneous or not). These leaderboards maintain state-of-the-art results and benchmarks on these tasks.

### Node Classification

#### Unsupervised Multi-label Node Classification

This leaderboard reports unsupervised multi-label node classification setting. we run all algorithms on several real-world datasets and report the sorted experimental results (Micro-F1 score with 90% labels as training data in L2 normalization logistic regression).

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

#### Semi-Supervised Node Classification with Attributes

This leaderboard reports the semi-supervised node classification under a transductive setting including several popular graph neural network methods.

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

#### Multiplex Node Classification

For multiplex node classification, we use macro F1 to evaluate models. We evaluate all models under the setting and datasets of GTN.

| Rank | Method                                                       |    DBLP     |    ACM    |   IMDB    |
| ---- | ------------------------------------------------------------ | :---------: | :-------: | :-------: |
| 1    | GTN [(Yun et al, NeurIPS'19)](https://arxiv.org/abs/1911.06455) | **92.03** | **90.85** | **59.24** |
| 2    | HAN [(Xiao et al, WWW'19)](https://arxiv.org/abs/1903.07293) |    91.21    |   87.25   |   53.94   |
| 3    | GCC [(Qiu et al, KDD'20)](http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD20-Qiu-et-al-GCC-GNN-pretrain.pdf) |    79.42    |   86.82   |    55.86  |
| 4    | PTE [(Tang et al, KDD'15)](https://arxiv.org/abs/1508.00200) |    78.65    |   87.44   |   48.91   |
| 5    | Metapath2vec [(Dong et al, KDD'17)](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) |    75.18    |   88.79   |   43.10   |
| 6    | Hin2vec [(Fu et al, CIKM'17)](https://dl.acm.org/doi/10.1145/3132847.3132953) |    74.31    |   84.66   |   44.04   |

### Link Prediction

#### Link Prediction

For link prediction, we adopt Area Under the Receiver Operating Characteristic Curve (ROC AUC), which represents the probability that vertices in a random unobserved link are more similar than those in a random nonexistent link. We evaluate these measures while removing 10 percents of edges on these dataset. We repeat our experiments for 10 times and report the results in order.

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

#### Multiplex Link Prediction

For multiplex link prediction, we adopt Area Under the Receiver Operating Characteristic Curve (ROC AUC). We evaluate these measures while removing 15 percents of edges on these dataset. We repeat our experiments for 10 times and report the three matrices in order.

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

### Graph Classification

This leaderboard reports the performance of graph classification methods. we run all algorithms on several datasets and report the sorted experimental results.

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

If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.

## Customization

### Submit Your State-of-the-art

If you have a well-performed algorithm and are willing to publish it, you can submit your implementation via [opening an issue](https://github.com/THUDM/cogdl/issues) or [join our slack group](https://join.slack.com/t/cogdl/shared_invite/enQtODgyMjY5MTY0NTY3LWQ5YTMwMWQzN2U2YTY0YWM2ZjhkNWUyZmE5ZmQyMTEyMGVkMzI0MjdlMGZlYmYzOWIwMTkyZGZmYTRjNGYxOGM). After evaluating its originality, creativity and efficiency, we will add your method's performance into our leaderboard.

### Add Your Own Dataset

If you have a unique and interesting dataset and are willing to publish it, you can submit your dataset via opening an issue in our repository or commenting on slack group, we will run all suitable methods on your dataset and update our leaderboard. 

### Implement Your Own Model

If you have a well-performed algorithm and are willing to implement it in our toolkit to help more people, you can create a pull request,  detailed information can be found [here](https://help.github.com/en/articles/creating-a-pull-request). 
