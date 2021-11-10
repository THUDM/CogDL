# CogDL examples for ogbn-arxiv

CogDL implementation of VRGCN for [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv):

> Jianfei Chen, Jun Zhu, Le Song. Stochastic Training of Graph Convolutional Networks with Variance Reduction. [Paper in arXiv](https://arxiv.org/abs/1710.10568). In ICML'2018.

Requires CogDL 0.5-alpha0 or later versions.


## Training & Evaluation

```
# Run with model with default config
python main.py
```
For more hyper-parameters, please find them in the `main.py`.

## Results

Here are the results over 10 runs which are comparable with OGB official results reported in the leaderboard.

|             Method              |  Test Accuracy  | Validation Accuracy | #Parameters |
|:-------------------------------:|:---------------:|:-------------------:|:-----------:|
|              VRGCN              | 0.7224 ± 0.0042 |   0.7260 ± 0.0030   |    44,328   |
