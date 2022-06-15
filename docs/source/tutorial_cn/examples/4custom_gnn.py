"""
自定义GNN
====================

"""

# %%
# 用CogDL 中的 GNN layers定义模型
# -----------------------------------
#  CogDL 在 cogdl.layers 中实现了流行的 GNN 层，它们可以作为模块来帮助您设计新的 GNN。以下是我们在 CogDL 中实现 Jumping Knowledge Network (JKNet) 的 GCNLayer 方法示例。 JKNet 收集所有层的输出并将它们连接在一起来获得结果：

import torch


from cogdl.layers import GCNLayer
from cogdl.models import BaseModel

class JKNet(BaseModel):
    def __init__(self, in_feats, out_feats, hidden_size, num_layers):
        super(JKNet, self).__init__()
        shapes = [in_feats] + [hidden_size] * num_layers
        self.layers = nn.ModuleList([
            GCNLayer(shapes[i], shapes[i+1])
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size * num_layers, out_feats)

    def forward(self, graph):
        # symmetric normalization of adjacency matrix
        graph.sym_norm()
        h = graph.x
        out = []
        for layer in self.layers:
            h = layer(graph,h)
            out.append(h)
        out = torch.cat(out, dim=1)
        return self.fc(out)

# %%
# 定义你的 GNN 模块
# ---------------------
# 在大多数情况下，您可以使用新的消息传播和聚合方案构建层模块。这里的代码片段展示了如何在 CogDL 中使用 Graph 和高效的稀疏矩阵算子来实现 GCNLayer。

import torch
from cogdl.utils import spmm

class GCNLayer(torch.nn.Module):
    """
    Args:
        in_feats: int
            Input feature size
        out_feats: int
            Output feature size
    """
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.fc = torch.nn.Linear(in_feats, out_feats)

    def forward(self, graph, x):
        h = self.fc(x)
        h = spmm(graph, h)
        return h

# %%
# 将自定义的GNN模型与Cogdl一起使用
# ------------------------------------
# 现在您已经定义了自己的 GNN，您可以使用 CogDL 中的数据集/任务来立即训练和评估模型的性能。

from cogdl import experiment
from cogdl.datasets import build_dataset_from_name
data = build_dataset_from_name("cora")[0]
# Use the JKNet model as defined above
model = JKNet(data.num_features, data.num_classes, 32, 4)
experiment(model=model, dataset="cora", mw="node_classification_mw", dw="node_classification_dw")