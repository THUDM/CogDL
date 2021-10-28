# CogDL examples for ogbn-arxiv

CogDL implementation of GCN and SAGE for [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv). 

Requires CogDL 0.5-alpha0 or later versions.


## Training & Evaluation

```
# Run with gcn model with default config
python gnn.py

# Run with sage model with default config
python gnn.py --model sage
```
For more hyper-parameters, please find them in the `gnn.py`.

## Results

Here are the results over 10 runs which are comparable with OGB official results reported in the leaderboard.

|             Method              |  Test Accuracy  | Validation Accuracy | #Parameters |
|:-------------------------------:|:---------------:|:-------------------:|:-----------:|
|               GCN               | 0.7168 ± 0.0030 |   0.7274 ± 0.0018   |   110,120   |
|            GraphSAGE            | 0.7224 ± 0.0014 |   0.7336 ± 0.0011   |   218,664   |
