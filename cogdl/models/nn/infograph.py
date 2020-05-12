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

    def forward(self, data, **kwargs):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            # print(out.shape) : [num_node x dim]
            feat_map.append(out)

        out = self.set2set(out, data.batch)
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
                mlp =  GINMLP(hidden_dim, hidden_dim, hidden_dim, num_mlp_layers, use_bn=True)
            self.gnn_layers.append(GINLayer(
               mlp, eps=0, train_eps=True
            ))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        if pooling == 'sum':
            self.pooling = scatter_add
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, batch):
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
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--target", dest='target', type=int, default=0,
                            help='')
        parser.add_argument("--train-num", dest='train_num', type=int, default=5000)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--use-unsupervised", dest="use_unsup", action="store_true")
        parser.add_argument("--epochs", type=int, default=40)
        parser.add_argument("--nn", type=bool, default=True)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.use_unsup
        )

    @classmethod
    def split_dataset(cls, dataset, args):
        if args.dataset == "QM9":
            test_dataset = dataset[:10000]
            val_dataset = dataset[10000:20000]
            train_dataset = dataset[20000:20000 + args.train_num]
            return DataLoader(train_dataset, batch_size=args.batch_size), DataLoader(val_dataset, batch_size=args.batch_size),\
                   DataLoader(test_dataset, batch_size=args.batch_size)
        else:
            test_index = random.sample(range(len(dataset)), len(dataset) // 10)
            train_index = [x for x in range(len(dataset)) if x not in test_index]

            train_dataset = [dataset[i] for i in train_index]
            test_dataset = [dataset[i] for i in test_index]
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            return train_loader, test_loader, test_loader

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
            self.unsup_encoder = SUPEncoder(in_feats, hidden_dim, num_layers)
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

    def forward(self, x, edge_index=None, batch=None, label=None):
        if self.unsup:
            return self.unsup_forward(x, edge_index, batch)
        else:
            return self.sup_forward(x, edge_index, batch, label)

    def sup_forward(self, x, edge_index=None, batch=None, label=None):
        node_feat, graph_feat = self.sem_encoder(x, edge_index, batch)
        node_feat = F.relu(self.sem_fc1(node_feat))
        node_feat = self.sem_fc2(node_feat)
        prediction = F.softmax(node_feat, dim=1)
        if label is not None:
            loss = self.sup_loss(prediction, label)
            loss += self.unsup_loss(x, edge_index, batch)
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

        num_graphs = sem_encode.shape[1]
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