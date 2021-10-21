import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import ModelWrapper
from cogdl.models.nn.mlp import MLP
from cogdl.data import DataLoader
from cogdl.wrappers.tools.wrapper_utils import evaluate_graph_embeddings_using_svm


class InfoGraphModelWrapper(ModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--sup", action="store_true")
        # fmt: on

    def __init__(self, model, optimizer_cfg, sup=False):
        super(InfoGraphModelWrapper, self).__init__()
        self.model = model
        hidden_size = optimizer_cfg["hidden_size"]
        model_num_layers = model.num_layers
        self.local_dis = FF(model_num_layers * hidden_size, hidden_size)
        self.global_dis = FF(model_num_layers * hidden_size, hidden_size)

        self.optimizer_cfg = optimizer_cfg
        self.sup = sup
        self.criterion = torch.nn.MSELoss()

    def train_step(self, batch):
        if self.sup:
            pred = self.model.sup_forward(batch, batch.x)
            loss = self.sup_loss(pred, batch)
        else:
            graph_feat, node_feat = self.model.unsup_forward(batch, batch.x)
            loss = self.unsup_loss(graph_feat, node_feat, batch.batch)
        return loss

    def test_step(self, dataset):
        device = self.device
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                preds.append(self.model(batch.to(device)))
        preds = torch.cat(preds).cpu().numpy()
        labels = np.array([g.y.item() for g in dataset])
        result = evaluate_graph_embeddings_using_svm(preds, labels)

        self.note("test_metric", result["acc"])
        self.note("std", result["std"])

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.global_dis.parameters()},
                {"params": self.local_dis.parameters()},
            ],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

    def sup_loss(self, pred, batch):
        pred = F.softmax(pred, dim=1)
        loss = self.criterion(pred, batch)
        loss += self.unsup_loss(batch.x, batch.edge_index, batch.batch)[1]
        return loss

    def unsup_loss(self, graph_feat, node_feat, batch):
        local_encode = self.local_dis(node_feat)
        global_encode = self.global_dis(graph_feat)

        num_graphs = graph_feat.shape[0]
        num_nodes = node_feat.shape[0]

        pos_mask = torch.zeros((num_nodes, num_graphs)).to(batch.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(batch.device)
        for nid, gid in enumerate(batch):
            pos_mask[nid][gid] = 1
            neg_mask[nid][gid] = 0
        glob_local_mi = torch.mm(local_encode, global_encode.t())
        loss = self.mi_loss(pos_mask, neg_mask, glob_local_mi, num_nodes, num_nodes * (num_graphs - 1))
        return loss

    @staticmethod
    def mi_loss(pos_mask, neg_mask, mi, pos_div, neg_div):
        pos_mi = pos_mask * mi
        neg_mi = neg_mask * mi

        pos_loss = (-math.log(2.0) + F.softplus(-pos_mi)).sum()
        neg_loss = (-math.log(2.0) + F.softplus(-neg_mi) + neg_mi).sum()
        # pos_loss = F.softplus(-pos_mi).sum()
        # neg_loss = (F.softplus(neg_mi)).sum()
        # pos_loss = pos_mi.sum()
        # neg_loss = neg_mi.sum()
        return pos_loss / pos_div + neg_loss / neg_div


class FF(nn.Module):
    r"""Residual MLP layers.

    ..math::
        out = \mathbf{MLP}(x) + \mathbf{Linear}(x)

    Paramaters
    ----------
    in_feats : int
        Size of each input sample
    out_feats : int
        Size of each output sample
    """

    def __init__(self, in_feats, out_feats):
        super(FF, self).__init__()
        self.block = MLP(in_feats, out_feats, out_feats, num_layers=3)
        self.shortcut = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        return F.relu(self.block(x)) + self.shortcut(x)
