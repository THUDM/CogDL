"""
图简介
=========================

"""
# %%
# 在Cogdl中表示图
# ------------------------

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
# 如何构建 mini-batch graphs
# ----------------------------------
# 在节点分类中，所有操作都在一个图中。但是在像图分类这样的任务中，我们需要用 mini-batch 处理很多图。图分类的数据集包含可以使用索引访问的图，例如data [2]。为了支持小批量训练/推理，CogDL 将一批中的图组合成一个完整的图，其中邻接矩阵形成稀疏块对角矩阵，其他的（节点特征、标签）在节点维度上连接。 这个过程由由cogdl.data.Dataloader来处理。
from cogdl.data import DataLoader
from cogdl.datasets import build_dataset_from_name

dataset = build_dataset_from_name("mutag")

print(dataset[0])

loader = DataLoader(dataset, batch_size=8)
for batch in loader:
    model(batch)

# %%
# 如何进行全局池化对每个图中节点的特征进行求和
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
# 如何编辑一个graph?
# ------------------------------
# 在某些设置中，可以更改edges。在这种情况下，我们需要在保留原始图的同时生成计算图。CogDL 提供了 graph.local_graph 来设置local scape，任何out-of-place 操作都不会反映到原始图上。但是， in-place操作会影响原始图形。
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
