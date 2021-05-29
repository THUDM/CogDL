# Simple-HGN

Simple-HGN code for heterogeneous node classification in cogdl [leaderboard](../../cogdl/tasks/README.md).

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --seed 0 1 2 3 4 -t heterogeneous_node_classification -dt gtn-acm -m simple_hgn --lr 0.001
CUDA_VISIBLE_DEVICES=0 python run.py --seed 0 1 2 3 4 -t heterogeneous_node_classification -dt gtn-dblp -m simple_hgn --lr 0.001
CUDA_VISIBLE_DEVICES=0 python run.py --seed 0 1 2 3 4 -t heterogeneous_node_classification -dt gtn-imdb -m simple_hgn --lr 0.001
```
