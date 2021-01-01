Link Prediction
===============

In this tutorial, we will introduce a important link prediction.
Overall speaking, the link prediction in CogDL can be divided into 3 types.

1. Network embeddings based link prediction(`HomoLinkPrediction`). All unsupervised network embedding methods supports this task for homogenous graphs without node features.
2. Knowledge graph completion(`KGLinkPrediction` and `TripleLinkPrediction`), including knowledge embedding methods(TransE, DistMult) and GNN base methods(RGCN and CompGCN).
3. GNN base homogenous graph link prediction(`GNNHomoLinkPrediction`). Theoretically, all GNN models works.


+-------------------------------+----------------------------------+
|                               | Models                           |
+===============================+==================================+
|Network embeddings methods     | DeepWalk, LINE, Node2Vec, ProNE  |
|                               | NetMF, NetSMF, SDNE, Hope        |
+-------------------------------+----------------------------------+
|Knowledge graph completion     | TransE, DistMult, RotatE,        |
|                               | RGCN, CompGCN                    |
+-------------------------------+----------------------------------+
| GNN methods                   | GCN and all the other GNN methods|
+-------------------------------+----------------------------------+

To implement a new GNN model for link prediction, just implement `link_prediction_loss` in the model which accepting thre parameters:

- Node features.
- Edge index.
- Labels. 0/1 for each item, indicating the edge exists in the graph or is a negative sample.

The overall implementation can be found at https://github.com/THUDM/cogdl/blob/master/cogdl/tasks/link_prediction.py