"""
模型训练
=========================

"""
# %%
# 自定义模型训练逻辑
# ------------------------
#cogdl 支持选择自定义训练逻辑，“数据-模型-训练”三部分在 CogDL 中是独立的，研究者和使用者可以自定义其中任何一部分，并复用其他部分，从而提高开发效率。现在您可以使用 Cogdl 中的模型和数据集来实现您的个性化需求。

import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl import experiment
from cogdl.datasets import build_dataset_from_name
from cogdl.layers import GCNLayer
from cogdl.models import BaseModel
class Gnn(BaseModel):
    def __init__(self, in_feats, hidden_size, out_feats, dropout):
        super(Gnn, self).__init__()
        self.conv1 = GCNLayer(in_feats, hidden_size)
        self.conv2 = GCNLayer(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout)
    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        h = F.relu(self.conv1(graph, self.dropout(h)))
        h = self.conv2(graph, self.dropout(h))
        return F.log_softmax(h, dim=1)

if __name__ == "__main__":
    dataset = build_dataset_from_name("cora")[0]
    model = Gnn(in_feats=dataset.num_features, hidden_size=64, out_feats=dataset.num_classes, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(dataset)
        loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    _, pred = model(dataset).max(dim=1)
    correct = float(pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item())
    acc = correct / dataset.test_mask.sum().item()
    print('The accuracy rate obtained by running the experiment with the custom training logic: {:.6f}'.format(acc))


# %%
# Experiment API
# --------------
# CogDL在训练上提供了更易于使用的 API ，即Experiment
from cogdl import experiment
experiment(model="gcn", dataset="cora")
#或者，您可以单独创建每个组件并使用CogDL 中的 build_dataset , build_model 来手动运行该过程。

from cogdl import experiment
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.options import get_default_args

args = get_default_args(model="gcn", dataset="cora")
dataset = build_dataset(args)
model = build_model(args)
experiment(model=model, dataset=dataset)

# %%
# 如何保存训练好的模型？
# --------------------------
experiment(model="gcn", dataset="cora", checkpoint_path="gcn_cora.pt")
# 当训练停止时，模型将保存在 gcn_cora.pt 中。如果你想从之前的checkpoint继续训练，使用不同的参数（如学习率、权重衰减等），保持相同的模型参数（如hidden size、模型层数），可以像下面这样做：
experiment(model="gcn", dataset="cora", checkpoint_path="gcn_cora.pt", resume_training=True)

# %%
# 如何保存embeddings?
# -------------------------
experiment(model="prone", dataset="blogcatalog", save_emb_path="./embeddings/prone_blog.npy")
# 以下代码片段评估我们在上面得到的embeddings：
experiment(
    model="prone",
    dataset="blogcatalog",
    load_emb_path="./embeddings/prone_blog.npy",
    num_shuffle=5,
    training_percents=[0.1, 0.5, 0.9]
)
