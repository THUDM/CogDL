# CogDL

CogDL is a graph representation learning toolkit that allows researchers and developers to easily train and compare baseline or custom models for node classification, link prediction and other tasks on graphs. It provides implementations of many popular models, including: non-GNN Baselines like Deepwalk, LINE, NetMF,  GNN Baselines like GCN, GAT, GraphSAGE.

Note that CogDL is **still actively under development**, so feedback and contributions are welcome.
Feel free to submit your contributions as a pull request.

CogDL features:

- Sparsification: fast network embedding on large-scale networks with tens of millions of nodes
- Arbitrary: dealing with different graph strucutures: attributed, multiplex and heterogeneous networks
- Parallel: parallel training of different seeds and different models on multiple GPUs and automatically reporting the result table
- Extensible: easily register new datasets, models, criterions and tasks

## Getting Started

## Requirements and Installation

- PyTorch version >= 1.0.0
- Python version >= 3.6

Please follow the instructions here to install PyTorch: https://github.com/pytorch/pytorch#installation.
Install other dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can use `python train.py --task example_task --dataset example_dataset --model example_method` to run example_method on example_data and evaluate it via example_task.

### General parameters

- --task, Downsteam tasks to evaluate representation like node_classification, unsupervised_node_classification, link_prediction
- --dataset, Dataset name to run, can be a list of datasets with space like `cora citeseer ppi`. Supported datasets including
'cora', 'citeseer', 'pumbed', 'PPI', 'wikipedia', 'blogcatalog', 'dblp', 'flickr'
- --model, Model name to run, can be a list of models like `deepwalk line prone`. Supported datasets including
'gcn', 'gat', 'graphsage', 'deepwalk', 'node2vec', 'hope', 'grarep', 'netmf', 'netsmf', 'prone'

For example, if you want to run Deepwalk, Line, Netmf on Wikipedia with node classification task, with 5 different seeds:

```bash
$ python train.py --task unsupervised_node_classification --dataset wikipedia --model line netmf --seed 0 1 2 3 4
```

Expected output:

| Variant                | Micro-F1 0.1   | Micro-F1 0.3   | Micro-F1 0.5   | Micro-F1 0.7   | Micro-F1 0.9   |
|------------------------|----------------|----------------|----------------|----------------|----------------|
| ('wikipedia', 'line')  | 0.4069±0.0011  | 0.4071±0.0010  | 0.4055±0.0013  | 0.4054±0.0020  | 0.4080±0.0042  |
| ('wikipedia', 'netmf') | 0.4551±0.0024  | 0.4932±0.0022  | 0.5046±0.0017  | 0.5084±0.0057  | 0.5125±0.0035  |

If you want to run parallel experiments on your server with multiple GPUs like multiple models gcn, gat on multiple datasets Cora, Citeseer with node classification task:

To enable efficient graph convolution on GPU, we require `pytorch_geometric`. Please install dependencies here https://github.com/rusty1s/pytorch_geometric/#installation.

```bash
$ python train.py --task node_classification --dataset cora citeseer --model gcn gat --device-id 0 1 2 3 --seed 0
```

Expected output:

| Variant             | Acc           |
|---------------------|---------------|
| ('cora', 'gcn')     | 0.8060±0.0000 |
| ('cora', 'gat')     | 0.8290±0.0000 |
| ('citeseer', 'gcn') | 0.7070±0.0000 |
| ('citeseer', 'gat') | 0.7110±0.0000 |

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

If you have a well-perform algorithm and are willing to public it, you can submit your implementation via [opening an issue](https://github.com/THUDM/cogdl/issues) or [join our slack group](https://join.slack.com/t/cogdl/shared_invite/enQtNjk1ODE4MjEyNDg2LTE3M2Y2N2QzNWJkYzcxNDMzYjZjYmY0YzlmMjYzZDliZTFiMGU3N2YzYWViNmVmNjk4OTY3YjYzODMzMDQ2ZGQ). After evaluating its originality, creativity and efficiency, we will add your method's performance into our leaderboard.

### Add Your Own Dataset

If you have a unique and interesting and are willing to public it, you can submit your dataset via opening an issue in our repository or commenting on slack group, we will run all suitable methods on your dataset and update our leaderboard. 

### Implement Your Own Model

If you have a well-perform algorithm and are willing to implement it in our toolkit to help more people, you can create a pull request,  detail information can be found [here](https://help.github.com/en/articles/creating-a-pull-request). 

## Citing

If you find *CogDL* is useful for your research, please consider citing the following papers:

```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}

@inproceedings{perozzi2014deepwalk,
  title={Deepwalk: Online learning of social representations},
  author={Perozzi, Bryan and Al-Rfou, Rami and Skiena, Steven},
  booktitle={Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={701--710},
  year={2014},
  organization={ACM}
}

@inproceedings{tang2015line,
  title={Line: Large-scale information network embedding},
  author={Tang, Jian and Qu, Meng and Wang, Mingzhe and Zhang, Ming and Yan, Jun and Mei, Qiaozhu},
  booktitle={Proceedings of the 24th International Conference on World Wide Web},
  pages={1067--1077},
  year={2015},
  organization={ACM}
}

@inproceedings{grover2016node2vec,
  title={node2vec: Scalable feature learning for networks},
  author={Grover, Aditya and Leskovec, Jure},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={855--864},
  year={2016},
  organization={ACM}
}

@inproceedings{cao2015grarep,
  title={Grarep: Learning graph representations with global structural information},
  author={Cao, Shaosheng and Lu, Wei and Xu, Qiongkai},
  booktitle={Proceedings of the 24th ACM International on Conference on Information and Knowledge Management},
  pages={891--900},
  year={2015},
  organization={ACM}
}

@inproceedings{Ou2016Asymmetric,
  title={Asymmetric Transitivity Preserving Graph Embedding},
  author={Ou, Mingdong and Cui, Peng and Pei, Jian and Zhang, Ziwei and Zhu, Wenwu},
  booktitle={The  ACM SIGKDD International Conference},
  pages={1105-1114},
  year={2016},
}

@inproceedings{qiu2018network,
  title={Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec},
  author={Qiu, Jiezhong and Dong, Yuxiao and Ma, Hao and Li, Jian and Wang, Kuansan and Tang, Jie},
  booktitle={Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining},
  pages={459--467},
  year={2018},
  organization={ACM}
}

@inproceedings{qiu2019netsmf,
  title={NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization},
  author={Qiu, Jiezhong and Dong, Yuxiao and Ma, Hao and Li, Jian and Wang, Chi and Wang, Kuansan and Tang, Jie},
  booktitle={The World Wide Web Conference},
  pages={1509--1520},
  year={2019},
  organization={ACM}
}

@article{zhang2018spectral,
  title={Spectral Network Embedding: A Fast and Scalable Method via Sparsity},
  author={Zhang, Jie and Wang, Yan and Tang, Jie},
  journal={arXiv preprint arXiv:1806.02623},
  year={2018}
}

@article{kipf2016semi,
  title={Semi-supervised classification with graph convolutional networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv:1609.02907},
  year={2016}
}

@article{Velickovic:17GAT,
  author    = {Petar Velickovic and
  Guillem Cucurull and
  Arantxa Casanova and
  Adriana Romero and
  Pietro Li{\`{o}} and
  Yoshua Bengio},
  title     = {Graph Attention Networks},
  journal   = {arXiv:1710.10903},
  year      = {2017},
}

@inproceedings{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  booktitle={NIPS'17},
  pages={1025--1035},
  year={2017}
}

@article{chen2018fastgcn,
  title={FastGCN: fast learning with graph convolutional networks via importance sampling},
  author={Chen, Jie and Ma, Tengfei and Xiao, Cao},
  journal={arXiv:1801.10247},
  year={2018}
}

@article{zou2019dimensional,
  title={Dimensional Reweighting Graph Convolutional Networks},
  author={Zou, Xu and Jia, Qiuye and Zhang, Jianwei and Zhou, Chang and Yang, Hongxia and Tang, Jie},
  journal={arXiv preprint arXiv:1907.02237},
  year={2019}
}
```
