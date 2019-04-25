# Cognitive Graph

Cognitive Graph is a graph representation learning toolkit based on [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/) that allows researchers and developers to train baseline or custom models for node classification, link prediction and other tasks on graphs.
It provides implementations of several models, including:

# Overview

- **Baselines**
  - [Kipf and Welling (2016): Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
  - [Hamilton et al. (2017): Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
  - [Veličković et al. (2017): Graph Attention Networks](https://arxiv.org/abs/1710.10903)
  - **_New_** [Ding et al. (2018): Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130)
  - **_New_** [Han et al. (2019): GroupRep: Unsupervised Structural Representation Learning for Groups in Networks](https://www.overleaf.com/read/nqxjtkmmgmff)
  - **_New_** [Zhang et al. (2019): Revisiting Graph Convolutional Networks: Neighborhood Aggregation and Network Sampling](https://www.overleaf.com/read/xzykmvhxjmxy)
  - **_New_** [Zhang et al. (2019): Co-training Graph Convolutional Networks with Network Redundancy](https://www.overleaf.com/read/fbhqqgzqgmyn)
- **Sparse**
  - **_New_** [Qiu et al. (2019): NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization](http://keg.cs.tsinghua.edu.cn/jietang/publications/www19-Qiu-et-al-NetSMF-Large-Scale-Network-Embedding.pdf)
  - **_New_** [Zhang et al. (2019): ProNE: Fast and Scalable Network Representation Learning](https://www.overleaf.com/read/dhgpkmyfdhnj)
- **QA**
  - **_New_** [Ding et al. (2019): Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://www.overleaf.com/8685337329vkbmkgckzfhk)
- **Heterogeneous**
  - **_New_** [Cen et al. (2019): Representation Learning for Attributed Multiplex Heterogeneous Network](https://www.overleaf.com/read/cfcyvptkzvjh)
- **Dynamic**
  - **_New_** [Zhao et al. (2019): Large Scale Evolving Graphs with Burst Detection](https://www.overleaf.com/4361782256sqswxgnmwbrs)

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

# Examples and pre-trained models

We have more detailed READMEs to reproduce results from specific papers:

- [Ding et al. (2019): Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://github.com/THUcqb/cognitive_graph/blob/master/examples/cogqa/README.md)

We also provide pre-trained models.

# License
