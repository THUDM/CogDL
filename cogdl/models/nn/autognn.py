import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import GATLayer, SELayer, GCNLayer, GCNIILayer
from cogdl import options, experiment
from cogdl.models import BaseModel, register_model
from cogdl.utils import spmm

def gcn_model(in_feats, hidden_size, num_layers, out_feats, dropout,residual, norm, activation):
    shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
            #if layers_connection == "skip-sum":
            #    self.layers_connection = nn.Linear(in_feats, out_feats)
            #else:
            #    self.layers_connection = None

    return nn.ModuleList(
        [
            GCNLayer(
                shapes[i],
                shapes[i + 1],
                dropout=dropout if i != num_layers - 1 else 0,
                residual=residual if i != num_layers - 1 else None,
                norm=norm if i != num_layers - 1 else None,
                activation=activation if i != num_layers - 1 else None,
            )
            for i in range(num_layers)
        ]
    )
    
def gat_model(in_feats, hidden_size, out_feats, nhead, attn_drop, alpha, residual, norm, num_layers, dropout, last_nhead):

    #if layers_connection == "skip-sum":
    #    self.layers_connection = nn.Linear(in_feats, out_feats * nhead)
    #else:
    #    self.layers_connection = None
    layers = nn.ModuleList()
    #if dropout > 0.0: self.layers.append(nn.Dropout(dropout))
    layers.append(
        GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm)
    )
    if num_layers != 1: layers.append(nn.ELU())
    for i in range(num_layers - 2):
        if dropout > 0.0 : layers.append(nn.Dropout(dropout))
        layers.append(
            GATLayer(
                hidden_size * nhead,
                hidden_size,
                nhead=nhead,
                attn_drop=attn_drop,
                alpha=alpha,
                residual=residual,
                norm=norm,
            )
        )
        layers.append(nn.ELU())
    
    if dropout > 0.0 : layers.append(nn.Dropout(p=dropout))
    layers.append(
        GATLayer(
            hidden_size * nhead,
            out_feats,
            attn_drop=attn_drop,
            alpha=alpha,
            nhead=last_nhead,
            residual=False,
        )
    )

    return layers

def grand_model(in_feats, hidden_size,out_feats, dropout, dropout2,norm):
    layers = nn.ModuleList()
    if norm == "batchnorm": layers.append(nn.BatchNorm1d(in_feats))
    layers.append(nn.Dropout(p=dropout))#dropout=inputdropout
    layers.append(nn.Linear(in_feats, hidden_size))
    layers.append(nn.ReLU())
    if norm == "batchnorm": layers.append(nn.BatchNorm1d(hidden_size))
    layers.append(nn.Dropout(p=dropout2))#dropout2
    layers.append(nn.Linear(hidden_size, out_feats))

    return layers


def gcnii_model(in_feats,hidden_size,out_feats,dropout,num_layers,alpha,lmbda,residual):
    layers = nn.ModuleList()
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_feats, hidden_size))
    layers.append(nn.ReLU())
    for i in range(num_layers):
        layers.append(nn.Dropout(p=dropout))
        layers.append(GCNIILayer(hidden_size, alpha, math.log(lmbda / (i + 1) + 1), residual))
        layers.append(nn.ReLU())
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_size, out_feats))

    return layers

def drgat_model(num_features,hidden_size,num_classes,dropout,num_heads):
    layers = nn.ModuleList()
    layers.append(nn.Dropout(p=dropout))
    layers.append(SELayer(num_features, se_channels=int(np.sqrt(num_features))))
    layers.append(GATLayer(num_features, hidden_size, nhead=num_heads, attn_drop=dropout))
    layers.append(nn.ELU())
    layers.append(nn.Dropout(p=dropout))
    layers.append(SELayer(hidden_size * num_heads, se_channels=int(np.sqrt(hidden_size * num_heads))))
    layers.append(GATLayer(hidden_size * num_heads, num_classes, nhead=1, attn_drop=dropout))
    layers.append(nn.ELU())

    return layers


@register_model("autognn")
class Autognn(BaseModel):
    """
        Args
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=8)
        parser.add_argument("--layers-type", type=str, default="gcn")
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.6)
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--attn-drop", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--last-nhead", type=int, default=1)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--dropoutn", type=float, default=0.5)


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.layers_type,
            args.dropout,
            args.activation,
            args.norm,
            args.residual,
            args.attn_drop,
            args.alpha,
            args.nhead,
            args.last_nhead,
            args.dropoutn,
        )    

    def __init__(
        self, 
        in_feats, 
        hidden_size, 
        out_feats, 
        num_layers, 
        layers_type, 
        dropout, 
        activation=None,
        norm=None, #复用use_bn
        residual=False,
        attn_drop=0.5,#复用dropnode
        alpha=0.2,
        nhead=8,#复用order
        last_nhead=1,
        dropoutn=0.5,#复用gcnii:lambda
        ):
        super(Autognn,self).__init__()
        
        self.dropout = dropout
        self.layers_type = layers_type
        if self.layers_type == "gcn":           
            self.layers = gcn_model(in_feats, hidden_size, num_layers, out_feats, dropout,residual, norm, activation)
            self.num_layers = num_layers    

        elif self.layers_type == "gat":         
            self.layers = gat_model(in_feats, hidden_size, out_feats, nhead, attn_drop, alpha, residual, norm, num_layers, dropout, last_nhead)
            self.num_layers = num_layers
            self.last_nhead = last_nhead
        elif self.layers_type == "grand":
            self.layers = grand_model(in_feats, hidden_size,out_feats, dropout, dropoutn,norm)
            self.dropnode_rate = attn_drop
            self.order = nhead
        elif self.layers_type == "gcnii":
            self.layers = gcnii_model(in_feats,hidden_size,out_feats,dropout,num_layers,alpha,dropoutn,residual)
        elif self.layers_type == "drgat":
            self.layers = drgat_model(in_feats,hidden_size,out_feats,dropout,nhead)

        self.autognn_parameters = list(self.layers.parameters())

    # def get_optimizer(self, args):
    #     return torch.optim.SGD(self.autognn_parameters,lr=args.lr,weight_decay=self.wd) if args.optimizer=="sgd" else torch.optim.Adam(self.autognn_parameters,lr=args.lr,weight_decay=self.wd)

    def drop_node(self, x):
        n = x.shape[0]
        drop_rates = torch.ones(n) * self.dropnode_rate
        if self.training:
            masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
            x = masks.to(x.device) * x

        else:
            x = x * (1.0 - self.dropnode_rate)
        return x

    def rand_prop(self, graph, x):
        x = self.drop_node(x)

        y = x
        for i in range(self.order):
            x = spmm(graph, x).detach_()
            y.add_(x)
        return y.div_(self.order + 1.0).detach_()

    def normalize_x(self, x):
        row_sum = x.sum(1)
        row_inv = row_sum.pow_(-1)
        row_inv.masked_fill_(row_inv == float("inf"), 0)
        x = x * row_inv[:, None]
        return x


    def forward(self, graph):
        if self.layers_type == "gcn":
            graph.sym_norm()
            h = graph.x
        elif self.layers_type == "gat":
            h = graph.x
        elif self.layers_type == "grand":
            graph.sym_norm()
            x = graph.x
            x = self.normalize_x(x)
            h = self.rand_prop(graph, x)
        elif self.layers_type == "gcnii":
            graph.sym_norm()
            h = graph.x
        elif self.layers_type == "drgat":
            h = graph.x
            


        for i, layer in enumerate(self.layers):
            
            if type(layer).__name__ == "GATLayer" or type(layer).__name__ == "GCNLayer":
                h = layer(graph, h)
            elif type(layer).__name__ == "GCNIILayer":
                h= layer(graph, h, init_h)
            else:
                h = layer(h)
            
            if i==2:
                init_h = h
        return h

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from cogdl.models.nn.pyg_srgcn import SrgcnHead, SrgcnSoftmaxHead
# from cogdl.utils.srgcn_utils import act_attention, act_map, act_normalization
# from cogdl import experiment
# from cogdl.models import BaseModel, register_model

# @register_model("autognn")
# class Autognn(BaseModel):
#     """
#         Args
#     """
#     @staticmethod
#     def add_args(parser):
#         parser.add_argument("--hidden-size", type=int, default=8)
#         parser.add_argument("--num-heads", type=int, default=8)
#         parser.add_argument("--dropout", type=float, default=0.5)
#         parser.add_argument("--node-dropout", type=float, default=0.5)
#         parser.add_argument("--alpha", type=float, default=0.2)
#         parser.add_argument("--lr", type=float, default=0.005)
#         parser.add_argument("--subheads", type=int, default=1)
#         parser.add_argument("--attention-type", type=str, default="node")
#         parser.add_argument("--activation", type=str, default="leaky_relu")
#         parser.add_argument("--nhop", type=int, default=1)
#         parser.add_argument("--normalization", type=str, default="row_uniform")


#     @classmethod
#     def build_model_from_args(cls, args):
#         return cls(
#             in_feats=args.num_features,
#             hidden_size=args.hidden_size,
#             out_feats=args.num_classes,
#             dropout=args.dropout,
#             node_dropout=args.node_dropout,
#             nhead=args.num_heads,
#             subheads=args.subheads,
#             alpha=args.alpha,
#             attention=args.attention_type,
#             activation=args.activation,
#             nhop=args.nhop,
#             normalization=args.normalization,
#         )    

#     def __init__(
#         self,
#         in_feats,
#         hidden_size,
#         out_feats,
#         attention,
#         activation,
#         nhop,
#         normalization,
#         dropout,
#         node_dropout,
#         alpha,
#         nhead,
#         subheads,
#         ):
#         super(Autognn,self).__init__()
#         attn_f = act_attention(attention)
#         activate_f = act_map(activation)
#         norm_f = act_normalization(normalization)
#         self.attentions = [
#             SrgcnHead(
#                 num_features=in_feats,
#                 out_feats=hidden_size,
#                 attention=attn_f,
#                 activation=activate_f,
#                 nhop=nhop,
#                 normalization=norm_f,
#                 subheads=subheads,
#                 dropout=dropout,
#                 node_dropout=node_dropout,
#                 alpha=alpha,
#                 concat=True,
#             )
#             for _ in range(nhead)
#         ]
#         for i, attention in enumerate(self.attentions):
#             self.add_module("attention_{}".format(i), attention)
#         self.out_att = SrgcnSoftmaxHead(
#             num_features=hidden_size * nhead * subheads,
#             out_feats=out_feats,
#             attention=attn_f,
#             activation=activate_f,
#             normalization=act_normalization("row_softmax"),
#             nhop=nhop,
#             dropout=dropout,
#             node_dropout=node_dropout,
#         )


#     def forward(self, graph):
#         x = torch.cat([att(graph, graph.x) for att in self.attentions], dim=1)
#         x = F.elu(x)
#         x = self.out_att(graph, x)
#         return x
            


if __name__ == "__main__":

    def func_search(trial):
        return {
            "dropout": trial.suggest_uniform("dropout", 0.5, 0.8),#intra-layer
            "norm": trial.suggest_categorical("norm", ["batchnorm", "layernorm"]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "gelu"]),
            "layers_type": trial.suggest_categorical("layers_type", ["gcn", "gat"]),
            "residual": trial.suggest_categorical("residual", [True, False]),#inter-layer
            "num_layers": trial.suggest_categorical("num_layers", [2, 4, 8]),
            "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),#config
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam"]),
            "max_epoch": trial.suggest_categorical("max_epoch", [500, 1000, 1500]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-4]),
        }


    experiment(task="node_classification", dataset="citeseer", model="autognn", seed=[1, 2], n_trials=100, func_search=func_search)