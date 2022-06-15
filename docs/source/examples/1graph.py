"""
Introduce of Graphs
=========================

"""
# %%
# How to represent a graph in CogDL
# -----------------------------------

import torch
from cogdl.data import Graph
edges = torch.tensor([[0,1],[1,3],[2,1],[4,2],[0,3]]).t()
x = torch.tensor([[-1],[0],[1],[2],[3]])
g = Graph(edge_index=edges,x=x) # equivalent to that above
print(g.row_indptr)

print(g.col_indices)

print(g.edge_weight)

print(g.num_nodes)

print(g.num_edges)

g.edge_weight = torch.rand(5)
print(g.edge_weight)

# %%
# How to construct mini-batch graphs
# ----------------------------------
# In node classification, all operations are in one single graph. But in tasks like graph classification, we need to deal with many graphs with mini-batch. Datasets for graph classification contains graphs which can be accessed with index, e.x. data[2]. To support mini-batch training/inference, CogDL combines graphs in a batch into one whole graph, where adjacency matrices form sparse block diagnal matrices and others(node features, labels) are concatenated in node dimension. cogdl.data.Dataloader handles the process.

from cogdl.data import DataLoader
from cogdl.datasets import build_dataset_from_name

dataset = build_dataset_from_name("mutag")

print(dataset[0])

loader = DataLoader(dataset, batch_size=8)
for batch in loader:
    model(batch)

# %%
# The following code snippet shows how to do global pooling to sum over features of nodes in each graph:
# --------------------------------------------------------------------------------------------------------
def batch_sum_pooling(x, batch):
    batch_size = int(torch.max(batch.cpu())) + 1
    res = torch.zeros(batch_size, x.size(1)).to(x.device)
    out = res.scatter_add_(
        dim=0,
        index=batch.unsqueeze(-1).expand_as(x),
        src=x
       )
    return out

    return out

# %%
# How to edit the graph?
# ------------------------------
# Changes can be applied to edges in some settings. In such cases, we need to generate a graph for calculation while keep the original graph. CogDL provides graph.local_graph to set up a local scape and any out-of-place operation will not reflect to the original graph. However, in-place operation will affect the original graph.

graph = build_dataset_from_name("cora")[0]
print(graph.num_edges)

with graph.local_graph():
    mask = torch.arange(100)
    row, col = graph.edge_index
    graph.edge_index = (row[mask], col[mask])
    print(graph.num_edges)

print(graph.num_edges)


print(graph.edge_weight)

with graph.local_graph():
    graph.edge_weight += 1
print(graph.edge_weight)
