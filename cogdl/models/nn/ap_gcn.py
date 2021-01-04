import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import BaseModel, register_model
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import MessagePassing
from torch.nn import ModuleList, Dropout, ReLU, Linear


class AdaptivePropagation(MessagePassing):
    def __init__(self, niter, h_size, **kwargs):
        super(AdaptivePropagation, self).__init__(aggr='add', **kwargs)

        self.niter = niter
        self.halt = Linear(h_size, 1)
        self.reg_params = list(self.halt.parameters())
        self.dropout = Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        self.halt.reset_parameters()
        x = (self.niter + 1) // 1
        b = math.log((1 / x) / (1 - (1 / x)))
        self.halt.bias.data.fill_(b)

    def forward(self, local_preds: torch.FloatTensor, edge_index):
        sz = local_preds.size(0)
        steps = torch.ones(sz).to(local_preds.device)
        sum_h = torch.zeros(sz).to(local_preds.device)
        continue_mask = torch.ones(sz, dtype=torch.bool).to(local_preds.device)
        x = torch.zeros_like(local_preds).to(local_preds.device)

        prop = self.dropout(local_preds)
        for _ in range(self.niter):

            old_prop = prop
            continue_fmask = continue_mask.type('torch.FloatTensor').to(local_preds.device)

            drop_edge_index, _ = dropout_adj(edge_index, training=self.training)
            drop_edge_index, _ = add_self_loops(drop_edge_index, num_nodes=sz)
            row, col = drop_edge_index
            deg = degree(col, sz, dtype=prop.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            drop_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            prop = self.propagate(drop_edge_index, x=prop, norm=drop_norm)

            h = torch.sigmoid(self.halt(prop)).t().squeeze()
            prob_mask = (((sum_h + h) < 0.99) & continue_mask).squeeze()
            prob_fmask = prob_mask.type('torch.FloatTensor').to(local_preds.device)

            steps = steps + prob_fmask
            sum_h = sum_h + prob_fmask * h

            final_iter = steps < self.niter

            condition = prob_mask & final_iter
            p = torch.where(condition, sum_h, 1 - sum_h)

            to_update = self.dropout(continue_fmask)[:, None]
            x = x + (prop * p[:, None] +
                     old_prop * (1 - p)[:, None]) * to_update

            continue_mask = continue_mask & prob_mask

            if (~continue_mask).all():
                break

        x = x / steps[:, None]

        return x, steps, (1 - sum_h)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


@register_model("ap_gcn")
class AP_GCN(BaseModel):
    """
    Model Name: Adaptive Propagation Graph Convolutional Network (AP-GCN)
    Paper link: https://arxiv.org/abs/2002.10306
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--train-ratio", type=float, default=0.7)
        parser.add_argument("--test-ratio", type=float, default=0.1)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--niter", type=int, default=10)
        parser.add_argument("--prop_penalty", type=float, default=0.005)
        parser.add_argument("--lr", type=float, default=0.001)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.dropout,
            args.niter,
            args.prop_penalty,
            args.weight_decay,
        )

    def __init__(self, in_feats, hidden_dim, out_feats, dropout, niter, prop_penalty, weight_decay):
        super(AP_GCN, self).__init__()

        num_features = [in_feats] + [hidden_dim] + [out_feats]

        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(nn.Linear(in_features, out_features))

        self.prop = AdaptivePropagation(niter, out_feats)
        self.prop_penalty = prop_penalty
        self.weight_decay = weight_decay
        self.layers = ModuleList(layers)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for layer in layers[1:] for p in layer.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.prop.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(self.dropout(x))

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)

        x, steps, reminders = self.prop(x, adj)
        return x, steps, reminders

    def node_classification_loss(self, data):
        x, steps, reminders = self.forward(data.x, data.edge_index)
        x = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(x[data.train_mask], data.y[data.train_mask])
        l2_reg = sum((torch.sum(param ** 2) for param in self.reg_params))
        loss += self.weight_decay / 2 * l2_reg + self.prop_penalty * (
                steps[data.train_mask] + reminders[data.train_mask]).mean()
        return loss

    def predict(self, data):
        x, _, _ = self.forward(data.x, data.edge_index)
        return x
