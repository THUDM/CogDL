# CogDL examples for ogbn-products

CogDL implementation of ClusterGCN (SAGE aggr) for [ogbn-products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products). 

Requires CogDL 0.5.1 or later versions.


## Training & Evaluation

```
# Run with sage model with default config
python gnn.py

# Run with sage model with custom config
python gnn.py --hidden-size 128
```
For more hyper-parameters, please find them in the `gnn.py`.

## Results

Here are the results over 10 runs which are comparable with OGB official results reported in the leaderboard.

|             Method              |  Test Accuracy  | Validation Accuracy | #Parameters |
|:-------------------------------:|:---------------:|:-------------------:|:-----------:|
|      ClusterGCN (SAGE aggr)     | 0.7906 ± 0.0032 |   0.9168 ± 0.0006   |   207,919   |
