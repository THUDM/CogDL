"""
Using Customized GNN
====================

"""

# %%
# GNN layers in CogDL to Define model
# -----------------------------------
#  CogDL has implemented popular GNN layers in cogdl.layers, and they can serve as modules to help design new GNNs. Here is how we implement Jumping Knowledge Network (JKNet) with GCNLayer in CogDL.
#  JKNet collects the output of all layers and concatenate them together to get the result:
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
# Define your GNN Module
# ---------------------
# In most cases, you may build a layer module with new message propagation and aggragation scheme. Here the code snippet shows how to implement a GCNLayer using Graph and efficient sparse matrix operators in CogDL.

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
# Use Custom models with CogDL
# ---------------------------
# Now that you have defined your own GNN, you can use dataset/task in CogDL to immediately train and evaluate the performance of your model.

data = build_dataset_from_name("cora")[0]
# Use the JKNet model as defined above
model = JKNet(data.num_features, data.num_classes, 32, 4)
experiment(model=model, dataset="cora", mw="node_classification_mw", dw="node_classification_dw")