"""
Model Training
=========================

"""
# %%
# Customized model training logic
# ----------------------------------
# cogdl supports the selection of custom training logic, you can use the models and data sets in Cogdl to achieve your personalized needs.

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
# CogDL provides a more easy-to-use API upon Trainer, i.e., experiment
from cogdl import experiment
experiment(model="gcn", dataset="cora")
# Or you can create each component separately and manually run the process using build_dataset, build_model in CogDL.

from cogdl import experiment
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.options import get_default_args

args = get_default_args(model="gcn", dataset="cora")
dataset = build_dataset(args)
model = build_model(args)
experiment(model=model, dataset=dataset)

# %%
# How to save trained model?
# --------------------------
experiment(model="gcn", dataset="cora", checkpoint_path="gcn_cora.pt")

# When the training stops, the model will be saved in gcn_cora.pt. If you want to continue the training from previous checkpoint with different parameters(such as learning rate, weight decay and etc.), keep the same model parameters (such as hidden size, model layers) and do it as follows:

experiment(model="gcn", dataset="cora", checkpoint_path="gcn_cora.pt", resume_training=True)

# %%
# How to save embeddings?
# ----------------------
experiment(model="prone", dataset="blogcatalog", save_emb_path="./embeddings/prone_blog.npy")

# In addition, CogDL supports evaluating node embeddings without training in different evaluation settings. The following code snippet evaluates the embedding we get above:

experiment(
    model="prone",
    dataset="blogcatalog",
    load_emb_path="./embeddings/prone_blog.npy",
    num_shuffle=5,
    training_percents=[0.1, 0.5, 0.9]
)
