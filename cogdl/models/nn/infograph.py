import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import NNConv, Set2Set

from .gin import GINLayer, GINMLP
from cogdl.data import DataLoader
from .. import BaseModel, register_model


class SUPEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_layers=1):
        super(SUPEncoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nnu = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nnu, aggr='mean', root_weight=False)
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        # self.lin1 = torch.nn.Linear(2 * dim, dim)
        # self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, x, edge_index, batch, edge_attr):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map.append(out)

        out = self.set2set(out, batch)
        return out, feat_map[-1]


class Encoder(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_layers=3, num_mlp_layers=2, pooling='sum'):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = GINMLP(in_feats, hidden_dim, hidden_dim, num_mlp_layers, use_bn=True)
            else:
                mlp = GINMLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, use_bn=True)
            self.gnn_layers.append(GINLayer(
               mlp, eps=0, train_eps=True
            ))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        if pooling == 'sum':
            self.pooling = scatter_add
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, batch, *args):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).cuda()
        layer_rep = []
        for i in range(self.num_layers):
            x = F.relu(self.bn_layers[i](self.gnn_layers[i](x, edge_index)))
            layer_rep.append(x)

        pooled_rep = [self.pooling(h, batch, 0) for h in layer_rep]
        node_rep = torch.cat(layer_rep, dim=1)
        graph_rep = torch.cat(pooled_rep, dim=1)
        return graph_rep, node_rep


class FF(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(FF, self).__init__()
        self.block = GINMLP(in_feats, out_feats, out_feats, num_layers=3 ,use_bn=False)
        self.shortcut = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        return F.relu(self.block(x)) + self.shortcut(x)


@register_model("infograph")
class InfoGraph(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-dim", type=int, default=64)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--target", dest='target', type=int, default=0,
                            help='')
        parser.add_argument("--train-num", dest='train_num', type=int, default=5000)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--unsup", dest="unsup", action="store_false")
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--nn", type=bool, default=True)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_dim,
            args.num_classes,
            args.num_layers,
            args.unsup
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        if args.dataset == "qm9":
            test_dataset = dataset[:10000]
            val_dataset = dataset[10000:20000]
            train_dataset = dataset[20000:20000 + args.train_num]
            return DataLoader(train_dataset, batch_size=args.batch_size), DataLoader(val_dataset, batch_size=args.batch_size),\
                   DataLoader(test_dataset, batch_size=args.batch_size)
        else:
            random.shuffle(dataset)
            train_size = int(len(dataset) * args.train_ratio)
            test_size = int(len(dataset) * args.test_ratio)
            bs = args.batch_size
            train_loader = DataLoader(dataset[:train_size], batch_size=bs)
            test_loader = DataLoader(dataset[-test_size:], batch_size=bs)
            if args.train_ratio + args.test_ratio < 1:
                valid_loader = DataLoader(dataset[train_size:-test_size], batch_size=bs)
            else:
                valid_loader = test_loader
            return train_loader, valid_loader, test_loader

    def __init__(self, in_feats, hidden_dim, out_feats, num_layers=3, unsup=True):
        super(InfoGraph, self).__init__()

        self.unsup = unsup
        self.emb_dim = hidden_dim
        self.out_feats = out_feats

        self.sem_fc1 = nn.Linear(num_layers*hidden_dim, hidden_dim)
        self.sem_fc2 = nn.Linear(hidden_dim, out_feats)

        if unsup:
            self.unsup_encoder = Encoder(in_feats, hidden_dim, num_layers)
            self.register_parameter("sem_encoder", None)
        else:
            self.unsup_encoder = Encoder(in_feats, hidden_dim, num_layers)
            self.sem_encoder = Encoder(in_feats, hidden_dim, num_layers)
        self._fc1 = FF(num_layers*hidden_dim, hidden_dim)
        self._fc2 = FF(num_layers*hidden_dim, hidden_dim)
        self.local_dis = FF(num_layers*hidden_dim, hidden_dim)
        self.global_dis = FF(num_layers*hidden_dim, hidden_dim)

        self.criterion = nn.MSELoss()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x, edge_index=None, batch=None, label=None, edge_attr=None):
        if self.unsup:
            return self.unsup_forward(x, edge_index, batch)
        else:
            return self.sup_forward(x, edge_index, batch, label, edge_attr)

    def sup_forward(self, x, edge_index=None, batch=None, label=None, edge_attr=None):
        node_feat, graph_feat = self.sem_encoder(x, edge_index, batch, edge_attr)
        node_feat = F.relu(self.sem_fc1(node_feat))
        node_feat = self.sem_fc2(node_feat)
        prediction = F.softmax(node_feat, dim=1)
        if label is not None:
            loss = self.sup_loss(prediction, label)
            loss += self.unsup_loss(x, edge_index, batch)[1]
            loss += self.unsup_sup_loss(x, edge_index, batch)
            return prediction, loss
        return prediction, None

    def unsup_forward(self, x, edge_index=None, batch=None):
        return self.unsup_loss(x, edge_index, batch)

    def sup_loss(self, prediction, label=None):
        sup_loss = self.criterion(prediction, label)
        return sup_loss

    def unsup_loss(self, x, edge_index=None, batch=None):
        graph_feat, node_feat = self.unsup_encoder(x, edge_index, batch)

        local_encode = self.local_dis(node_feat)
        global_encode = self.global_dis(graph_feat)

        num_graphs = graph_feat.shape[0]
        num_nodes = node_feat.shape[0]

        pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
        neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
        for nid, gid in enumerate(batch):
            pos_mask[nid][gid] = 1
            neg_mask[nid][gid] = 0
        glob_local_mi = torch.mm(local_encode, global_encode.t())
        loss = InfoGraph.mi_loss(pos_mask, neg_mask, glob_local_mi, num_nodes, num_nodes * (num_graphs-1))
        return graph_feat, loss

    def unsup_sup_loss(self, x, edge_index, batch):
        sem_g_feat, _ = self.sem_encoder(x, edge_index, batch)
        un_g_feat, _ = self.unsup_encoder(x, edge_index, batch)

        sem_encode = self._fc1(sem_g_feat)
        un_encode = self._fc2(un_g_feat)

        num_graphs = sem_encode.shape[0]
        pos_mask = torch.eye(num_graphs).cuda()
        neg_mask = 1 - pos_mask

        mi = torch.mm(sem_encode, un_encode.t())
        loss = InfoGraph.mi_loss(pos_mask, neg_mask, mi, pos_mask.sum(), neg_mask.sum())
        return loss

    @staticmethod
    def mi_loss(pos_mask, neg_mask, mi, pos_div, neg_div):
        pos_mi = pos_mask * mi
        neg_mi = neg_mask * mi
        # pos_loss = -(math.log(2.) - F.softplus(-pos_mi)).sum()
        # neg_loss = (-math.log(2.) + F.softplus(-neg_mi) + neg_mi).sum()
        pos_loss = F.softplus(-pos_mi).sum()
        neg_loss = (F.softplus(neg_mi)).sum()
        return pos_loss/pos_div + neg_loss/neg_div