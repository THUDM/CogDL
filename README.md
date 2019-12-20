![CogDL](docs/source/_static/cogdl-logo.png)
===

[![Build Status](https://travis-ci.org/THUDM/cogdl.svg?branch=master)](https://travis-ci.org/THUDM/cogdl)
[![Maintainability](https://api.codeclimate.com/v1/badges/d587092245542684c80b/maintainability)](https://codeclimate.com/github/THUDM/cogdl/maintainability)
[![Coverage Status](https://coveralls.io/repos/github/THUDM/cogdl/badge.svg?branch=master)](https://coveralls.io/github/THUDM/cogdl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/cogdl/badge/?version=latest)](https://cogdl.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/thudm/cogdl)](https://github.com/THUDM/cogdl/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

CogDL is a graph representation learning toolkit that allows researchers and developers to easily train and compare baseline or custom models for node classification, link prediction and other tasks on graphs. It provides implementations of many popular models, including: non-GNN Baselines like Deepwalk, LINE, NetMF,  GNN Baselines like GCN, GAT, GraphSAGE.

Note that CogDL is **still actively under development**, so feedback and contributions are welcome.
Feel free to submit your contributions as a pull request.

CogDL features:

- Sparsification: fast network embedding on large-scale networks with tens of millions of nodes
- Arbitrary: dealing with different graph structures: attributed, multiplex and heterogeneous networks
- Parallel: parallel training of different seeds and different models on multiple GPUs and automatically reporting the result table
- Extensible: easily register new datasets, models, criteria and tasks
- Supported tasks:
  - Node classification
  - Link prediction
  - Social influence prediction (coming)
  - Knowledge graph reasoning (coming)
  - Community detection (todo)
  - Graph classification (todo)
  - Combinatorial optimization on graphs (todo)

## Getting Started

## Requirements and Installation

- PyTorch version >= 1.0.0
- Python version >= 3.6

Please follow the instructions here to install PyTorch: https://github.com/pytorch/pytorch#installation.
Install other dependencies:

```bash
pip install -e .
```

## Usage

You can use `python scripts/train.py --task example_task --dataset example_dataset --model example_method` to run example_method on example_data and evaluate it via example_task.

### General parameters

- --task, Downstream tasks to evaluate representation like node_classification, unsupervised_node_classification, link_prediction
- --dataset, Dataset name to run, can be a list of datasets with space like `cora citeseer ppi`. Supported datasets including
'cora', 'citeseer', 'pumbed', 'PPI', 'wikipedia', 'blogcatalog', 'dblp', 'flickr'
- --model, Model name to run, can be a list of models like `deepwalk line prone`. Supported datasets including
'gcn', 'gat', 'graphsage', 'deepwalk', 'node2vec', 'hope', 'grarep', 'netmf', 'netsmf', 'prone'

For example, if you want to run Deepwalk, Line, Netmf on Wikipedia with node classification task, with 5 different seeds:

```bash
$ python scripts/train.py --task unsupervised_node_classification --dataset wikipedia --model line netmf --seed 0 1 2 3 4
```

Expected output:

| Variant                | Micro-F1 0.1   | Micro-F1 0.3   | Micro-F1 0.5   | Micro-F1 0.7   | Micro-F1 0.9   |
|------------------------|----------------|----------------|----------------|----------------|----------------|
| ('wikipedia', 'line')  | 0.4069±0.0011  | 0.4071±0.0010  | 0.4055±0.0013  | 0.4054±0.0020  | 0.4080±0.0042  |
| ('wikipedia', 'netmf') | 0.4551±0.0024  | 0.4932±0.0022  | 0.5046±0.0017  | 0.5084±0.0057  | 0.5125±0.0035  |

If you want to run parallel experiments on your server with multiple GPUs like multiple models gcn, gat on multiple datasets Cora, Citeseer with node classification task:

To enable efficient graph convolution on GPU, we require `pytorch_geometric`. Please install dependencies here https://github.com/rusty1s/pytorch_geometric/#installation.

```bash
$ python scripts/parallel_train.py --task node_classification --dataset cora --model pyg_gcn pyg_gat --device-id 0 1 --seed 0 1 2 3 4
```

Expected output:

| Variant             | Acc           |
|---------------------|---------------|
| ('cora', 'pyg_gcn') | 0.7922±0.0082 |
| ('cora', 'pyg_gat') | 0.8092±0.0055 |

### Specific parameters

for DeepWalk and node2vec:

- --walk-num, the number of random walks to start at each node; the default is 10;
- --walk-length, Length of walk start at each node. Default is 50;
- --worker, Number of parallel workers. Default is 10;
- --window-size, Window size of skip-gram model. Default is 10;
- --iteration, Number of iterations. Default is 10;
- --q, Parameter in node2vec. Default is 1.0;
- --p, Parameter in node2vec. Default is 1.0;

for LINE:

- --order, Order of proximity in LINE. Default is 3 for 1+2;
- --alpha, Initial earning rate of SGD. Default is 0.025;
- --batch-size, Batch size in SGD training process. Default is 100;
- --negative, Number of negative nodes in sampling. Default is 5;

for HOPE:

- --beta, Parameter of katz for HOPE. Default is 0.01;

for Grarep:

- --step, Number of matrix step in GraRep and ProNE. Default is 5;

for NetMF:

- --window-size, Window size of deepwalk matrix. Default is 10;
- --is-large, Large or small for NetMF;
- --negative, Number of negative nodes in sampling. Default is 5;
- --rank, Number of Eigenpairs in NetMF, default is 256;

for NetSMF:

- --window-size, Window size of approximate matrix. Default is 10;
- --negative, Number of negative nodes in sampling. Default is 5;
- --round, Number of round in NetSMF. Default is 100;
- --worker, Number of parallel workers. Default is 10;

for ProNE:

- --step, Number of items in the chebyshev expansion. Default is 5;
- --theta, Parameter of ProNE. Default is 0.5;
- --mu, Parameter of ProNE. Default is 0.2;

for GCN and DR-GCN:

- --hidden-size, The size of hidden layer. Default=16;
- --num-layers, The number of GCN layer. Default=2;
- --dropout, The dropout probability. Default=0.5;

for GAT and DR-GAT:

- --hidden-size, The size of hidden layer. Default=8;
- --num-heads, The number of heads in attention mechanism. Default=8;
- --dropout, The dropout probability. Default=0.6;

for Graphsage:

- --hidden-size, The size of hidden layer. Default=8;
- --num-layers, The number of Graphsage. Default=2;
- --sample-size, The List of number of neighbor samples for each node in Graphsage. Default=10, 10;
- --dropout, The dropout probability. Default=0.5;

If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.

## Customization

### Submit Your State-of-the-art

If you have a well-perform algorithm and are willing to publish it, you can submit your implementation via [opening an issue](https://github.com/THUDM/cogdl/issues) or [join our slack group](https://join.slack.com/t/cogdl/shared_invite/enQtODgyMjY5MTY0NTY3LWQ5YTMwMWQzN2U2YTY0YWM2ZjhkNWUyZmE5ZmQyMTEyMGVkMzI0MjdlMGZlYmYzOWIwMTkyZGZmYTRjNGYxOGM). After evaluating its originality, creativity and efficiency, we will add your method's performance into our leaderboard.

### Add Your Own Dataset

If you have a unique and interesting and are willing to publish it, you can submit your dataset via opening an issue in our repository or commenting on slack group, we will run all suitable methods on your dataset and update our leaderboard. 

### Implement Your Own Model

If you have a well-perform algorithm and are willing to implement it in our toolkit to help more people, you can create a pull request,  detailed information can be found [here](https://help.github.com/en/articles/creating-a-pull-request). 
