import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
from .gin import GINLayer, split_dataset_general
from cogdl.utils import batch_mean_pooling, batch_sum_pooling
from .. import BaseModel, register_model


class Encoder(nn.Module):
    r"""Encoder stacked with GIN layers

    Parameters
    ----------
    in_feats : int
        Size of each input sample.
    hidden_feats : int
        Size of output embedding.
    num_layers : int, optional
        Number of GIN layers, default: ``3``.
    num_mlp_layers : int, optional
        Number of MLP layers for each GIN layer, default: ``2``.
    pooling : str, optional
        Aggragation type, default : ``sum``.

    """

    def __init__(self, in_feats, hidden_dim, num_layers=3, num_mlp_layers=2, pooling="sum"):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = MLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, norm="batchnorm")
            self.gnn_layers.append(GINLayer(mlp, eps=0, train_eps=True))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        if pooling == "sum":
            self.pooling = batch_sum_pooling
        elif pooling == "mean":
            self.pooling = batch_mean_pooling
        else:
            raise NotImplementedError

    def forward(self, graph, x=None, *args):
        batch = graph.batch
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(batch.device)
        layer_rep = []
        for i in range(self.num_layers):
            x = F.relu(self.bn_layers[i](self.gnn_layers[i](graph, x)))
            layer_rep.append(x)

        pooled_rep = [self.pooling(h, batch) for h in layer_rep]
        node_rep = torch.cat(layer_rep, dim=1)
        graph_rep = torch.cat(pooled_rep, dim=1)
        return graph_rep, node_rep


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


@register_model("infograph")
class InfoGraph(BaseModel):
    r"""Implimentation of Infograph in paper `"InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation
     Learning via Mutual Information Maximization" <https://openreview.net/forum?id=r1lfF2NYvH>__. `

     Parameters
     ----------
     in_feats : int
        Size of each input sample.
    out_feats : int
        Size of each output sample.
    num_layers : int, optional
        Number of MLP layers in encoder, default: ``3``.
    unsup : bool, optional
        Use unsupervised model if True, default: ``True``.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--target", dest="target", type=int, default=0, help="")
        parser.add_argument("--train-num", dest="train_num", type=int, default=5000)
        parser.add_argument("--num-layers", type=int, default=1)
        parser.add_argument("--sup", dest="sup", action="store_true")
        parser.add_argument("--epoch", type=int, default=15)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.num_layers, args.sup)

    @classmethod
    def split_dataset(cls, dataset, args):
        return split_dataset_general(dataset, args)

    def __init__(self, in_feats, hidden_dim, out_feats, num_layers=3, sup=False):
        super(InfoGraph, self).__init__()

        self.sup = sup
        self.emb_dim = hidden_dim
        self.out_feats = out_feats

        self.sem_fc1 = nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.sem_fc2 = nn.Linear(hidden_dim, out_feats)

        if not sup:
            self.unsup_encoder = Encoder(in_feats, hidden_dim, num_layers)
            self.register_parameter("sem_encoder", None)
        else:
            self.unsup_encoder = Encoder(in_feats, hidden_dim, num_layers)
            self.sem_encoder = Encoder(in_feats, hidden_dim, num_layers)
        self._fc1 = FF(num_layers * hidden_dim, hidden_dim)
        self._fc2 = FF(num_layers * hidden_dim, hidden_dim)
        self.local_dis = FF(num_layers * hidden_dim, hidden_dim)
        self.global_dis = FF(num_layers * hidden_dim, hidden_dim)

        self.criterion = nn.MSELoss()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, batch):
        if self.sup:
            return self.sup_forward(batch, batch.x)
        else:
            return self.unsup_forward(batch, batch.x)

    def graph_classification_loss(self, batch):
        if self.sup:
            pred = self.sup_forward(batch, batch.x)
            loss = self.sup_loss(pred, batch)
        else:
            graph_feat, node_feat = self.unsup_forward(batch, batch.x)
            loss = self.unsup_loss(graph_feat, node_feat, batch.batch)
        return loss

    def sup_forward(self, batch, x):
        node_feat, graph_feat = self.sem_encoder(batch, x)
        node_feat = F.relu(self.sem_fc1(node_feat))
        node_feat = self.sem_fc2(node_feat)
        return node_feat

    def unsup_forward(self, batch, x):
        # return self.unsup_loss(x, edge_index, batch)
        graph_feat, node_feat = self.unsup_encoder(batch, x)
        if self.training:
            return graph_feat, node_feat
        else:
            return graph_feat

    def sup_loss(self, pred, batch):
        pred = F.softmax(pred, dim=1)
        loss = self.criterion(pred, batch)
        loss += self.unsup_loss(batch.x, batch.edge_index, batch.batch)[1]
        loss += self.unsup_sup_loss(batch, batch.x)
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
        loss = InfoGraph.mi_loss(pos_mask, neg_mask, glob_local_mi, num_nodes, num_nodes * (num_graphs - 1))
        return loss

    def unsup_sup_loss(self, batch, x):
        sem_g_feat, _ = self.sem_encoder(batch, x)
        un_g_feat, _ = self.unsup_encoder(batch, x)

        sem_encode = self._fc1(sem_g_feat)
        un_encode = self._fc2(un_g_feat)

        num_graphs = sem_encode.shape[0]
        pos_mask = torch.eye(num_graphs).to(x.device)
        neg_mask = 1 - pos_mask

        mi = torch.mm(sem_encode, un_encode.t())
        loss = InfoGraph.mi_loss(pos_mask, neg_mask, mi, pos_mask.sum(), neg_mask.sum())
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
