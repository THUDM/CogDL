# Cognitive Graph

Cognitive Graph is a graph representation learning toolkit based on [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/) that allows researchers and developers to train baseline or custom models for node classification, link prediction and other tasks on graphs.
It provides implementations of several models, including:

# Overview

- **Non-GNN Baselines**
  - [Perozzi et al. (2014): Deepwalk: Online learning of social representations](http://arxiv.org/abs/1403.6652)
  - [Tang et al. (2015): Line: Large-scale informa- tion network embedding](http://arxiv.org/abs/1503.03578)
  - [Grover and Leskovec. (2016): node2vec: Scalable feature learning for networks](http://dl.acm.org/citation.cfm?doid=2939672.2939754)
  - [Cao et al. (2015):Grarep: Learning graph representations with global structural information ](http://dl.acm.org/citation.cfm?doid=2806416.2806512)
  - [Ou et al. (2016): Asymmetric transitivity preserving graph em- bedding](http://dl.acm.org/citation.cfm?doid=2939672.2939751)
  - [Qiu et al. (2017): Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](http://arxiv.org/abs/1710.02971)
  - [Qiu et al. (2019): NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization](http://arxiv.org/abs/1710.02971)
  - [Zhang et al. (2019): Spectral Network Embedding: A Fast and Scalable Method via Sparsity](http://arxiv.org/abs/1806.02623)

- **GNN Baselines**
  - [Kipf and Welling (2016): Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
  - [Hamilton et al. (2017): Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
  - [Veličković et al. (2017): Graph Attention Networks](https://arxiv.org/abs/1710.10903)
  - **_New_! CIKM 2018** [Ding et al. (2018): Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130)
  - **_New_** [Han et al. (2019): GroupRep: Unsupervised Structural Representation Learning for Groups in Networks](https://www.overleaf.com/read/nqxjtkmmgmff)
  - **_New_** [Zhang et al. (2019): Revisiting Graph Convolutional Networks: Neighborhood Aggregation and Network Sampling](https://www.overleaf.com/read/xzykmvhxjmxy)
  - **_New_** [Zhang et al. (2019): Co-training Graph Convolutional Networks with Network Redundancy](https://www.overleaf.com/read/fbhqqgzqgmyn)
- **Sparse**
  - **_New!_ WWW 2019** [Qiu et al. (2019): NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization](http://keg.cs.tsinghua.edu.cn/jietang/publications/www19-Qiu-et-al-NetSMF-Large-Scale-Network-Embedding.pdf)
  - **_New!_ IJCAI 2019** [Zhang et al. (2019): ProNE: Fast and Scalable Network Representation Learning](https://www.overleaf.com/read/dhgpkmyfdhnj)
- **QA**
  - **_New!_ ACL 2019** [Ding et al. (2019): Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://arxiv.org/abs/1905.05460)
- **Heterogeneous**
  - **_New!_ KDD 2019** [Cen et al. (2019): Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/abs/1905.01669)
- **Dynamic**
  - **_New!_ IJCAI 2019** [Zhao et al. (2019): Large Scale Evolving Graphs with Burst Detection](https://www.overleaf.com/4361782256sqswxgnmwbrs)

Cognitive Graph features:

- sparsification: fast network embedding on large-scale networks with tens of millions of nodes
- cognitive: multi-hop question answering based on GNN and BERT
- arbitrary: dealing with different graph strucutures: attributed, multiplex and heterogeneous networks
- distributed: multi-GPU training on one machine or across multiple machines
- extensible: easily register new datasets, models, criterions and tasks

# Requirements and Installation

- PyTorch version >= 1.0.0
- Python version >= 3.6

Please follow the instructions here to install PyTorch and other dependencies: https://github.com/pytorch/pytorch#installation, https://github.com/rusty1s/pytorch_geometric/#installation

# Getting Started

```bash
$ python display_data.py --dataset cora
+-----------+----------+----------+-------------+------------+-----------------+
| Dataset   |   #nodes |   #edges |   #features |   #classes |   #labeled data |
|-----------+----------+----------+-------------+------------+-----------------|
| cora      |     2708 |    10556 |        1433 |          7 |             140 |
+-----------+----------+----------+-------------+------------+-----------------+
Sampled ego network saved to ./display.png .

$ python train.py --dataset cora --model gat --num-heads 8 --hidden-size 8 --dropout 0.6 --max-epoch 100 --lr 0.005 --weight-decay 5e-4
Epoch: 099, Train: 0.9786, Val: 0.8060: 100%|███████████████████████████| 100/100 [00:01<00:00, 66.77it/s]
Test accuracy = 0.826

$ python train.py --dataset cora --model gcn --num-layers 2 --hidden-size 32 --dropout 0.5 --max-epoch 100 --lr 0.01 --weight-decay 5e-4
Epoch: 099, Train: 0.9857, Val: 0.7900: 100%|██████████████████████████| 100/100 [00:00<00:00, 142.42it/s]
Test accuracy = 0.813
```

# Usage

You can use `python train.py --task example_task --dataset example_dataset --model example_method` to run example_method on example_data and evaluate it via example_task.

## General parameters

- --task, Downsteam tasks to evaluate representation like node_classification, unsupervised_node_classification, link_prediction
- --dataset, Dataset name to run, can be a list of datasets with space like `cora citeseer ppi`. Supported datasets including
'cora', 'citeseer', 'pumbed', 'PPI', 'wikipedia', 'blogcatalog', 'dblp', 'flickr'
- --model, Model name to run, can be a list of models like `deepwalk line prone`. Supported datasets including
'gcn', 'gat', 'deepwalk', 'node2vec', 'hope', 'grarep', 'netmf', 'netsmf', 'prone'


## Specific parameters
for DeepWalk and node2vec:
- --walk-num, the number of random walks to start at each node; the default is 10;
- --walk-length, Length of walk start at each node. Default is 50;
- --worker, Number of parallel workers. Default is 10;
- --window-size, Window size of skip-gram model. Default is 10;
- --q, Parameter in node2vec. Default is 1.0;
- --p, Parameter in node2vec. Default is 1.0;

for LINE:
- --order, Order of proximity in LINE. Default is 3 for 1+2;
- --alpha, Initial earning rate of SGD. Default is 0.025;
- --batch-size, Batch size in SGD training process. Default is 100;
- --negative, Number of negative node in sampling. Default is 5;

for HOPE:
- --beta, Parameter of katz for HOPE. Default is 0.01;

for Grarep:
- --step, Number of matrix step in GraRep and ProNE. Default is 5;

for NetMF:
- --window-size, Window size of deepwalk matrix. Default is 10;
- --is-large, Large or small for NetMF;
- --negative, Number of negative node in sampling. Default is 5;
- --rank, Number of Eigenpairs in NetMF, default is 256;

for NetSMF:
- --window-size, Window size of approximate matrix. Default is 10;
- --negative, Number of negative node in sampling. Default is 5;
- --round, Number of round in NetSMF. Default is 100;
- --worker, Number of parallel workers. Default is 10;

for ProNE:
- --step, Number of items in the chebyshev expansion. Default is 5;
- --theta, Parameter of ProNE. Default is 0.5;
- --mu, Parameter of ProNE. Default is 0.2;

for GCN:
- --hidden-size, The size of hidden layer. Default=16;
- --num-layers, The number of GCN layer. Default=2;
- --dropout, The dropout probability. Default=0.5;

for GAT:
- --hidden-size, The size of hidden layer. Default=8;
- --num-heads, The number of heads in attention mechanism. Default=8;
- --dropout, The dropout probability. Default=0.6;


If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.



# Examples and pre-trained models

We have more detailed READMEs to reproduce results from specific papers:

- [Ding et al. (2019): Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://github.com/THUcqb/cognitive_graph/blob/master/examples/cogqa/README.md)

We also provide pre-trained models.

# License
