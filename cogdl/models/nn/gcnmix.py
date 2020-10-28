import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .. import BaseModel, register_model


def mix_hidden_state(feat, target, train_index, alpha):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1
    permuted_index = train_index[torch.randperm(train_index.size(0))]
    feat[train_index] = lamb * feat[train_index] + (1 - lamb) * feat[permuted_index]
    return feat, target[train_index], target[permuted_index], lamb


def sharpen(prob, temperature):
    prob = torch.pow(prob, 1./temperature)
    row_sum = torch.sum(prob, dim=1).reshape(-1, 1)
    return prob / row_sum


def get_one_hot_label(labels, index):
    num_classes = int(torch.max(labels) + 1)
    target = torch.zeros(labels.shape[0], num_classes).to(labels.device)
    # target.scatter_(1, torch.unsqueeze(labels[index], 1), 1.0)
    target[index, labels[index]] = 1
    return target


def get_current_consistency_weight(final_consistency_weight, rampup_starts, rampup_ends, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    rampup_length = rampup_ends - rampup_starts
    rampup = 1.0
    epoch = epoch - rampup_starts
    if rampup_length != 0:
        current = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        rampup = float(np.exp(-5.0 * phase * phase))
    return final_consistency_weight * rampup


class BaseGNNMix(BaseModel):
    def __init__(self, in_feat, hidden_size, num_classes, k, temperature, alpha, dropout):
        super(BaseGNNMix, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.k = k
        self.temperature = temperature

        self.input_gnn = GCNConv(in_feat, hidden_size)
        self.hidden_gnn = GCNConv(hidden_size, num_classes)
        self.loss_f = nn.BCELoss()

    def forward(self, x, edge_index, label=None, train_index=None, mix_hidden=False):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.input_gnn(h, edge_index)
        h = F.relu(h)

        if mix_hidden:
            h, target, target_mix, lamb = mix_hidden_state(h, label, train_index, self.alpha)
            # h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.hidden_gnn(h, edge_index)
            return h, target, target_mix, lamb
        else:
            # h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.hidden_gnn(h, edge_index)
            return h

    def update_mix(self, data, vector_labels, train_index, opt):
        device = data.x.device
        train_unlabelled = torch.where(data.train_mask == False)[0].to(device)
        temp_labels = torch.zeros(self.k, vector_labels.shape[0], vector_labels.shape[1]).to(device)
        with torch.no_grad():
            for i in range(self.k):
                temp_labels[i, :, :] = self.predict_noise(data)

        target_labels = temp_labels.mean(dim=0)
        target_labels = sharpen(target_labels, self.temperature)
        vector_labels[train_unlabelled] = target_labels[train_unlabelled]
        sampled_unlabelled = torch.randint(0, train_unlabelled.shape[0], size=(train_index.shape[0], ))
        train_unlabelled = train_unlabelled[sampled_unlabelled]

        def get_loss(index):
            mix_logits, label, permuted_label, lamb = self.forward(data.x, data.edge_index, data.y, index, mix_hidden=True)
            mix_label = lamb * vector_labels[label] + (1 - lamb) * vector_labels[permuted_label]
            temp_loss = self.loss_f(F.softmax(mix_logits[index], -1), mix_label)
            return temp_loss
        sup_loss = get_loss(train_index)
        unsup_loss = get_loss(train_unlabelled)

        mixup_weight = get_current_consistency_weight(opt["final_consistency_weight"], opt["rampup_starts"],
                                                      opt["rampup_ends"], opt["epoch"])
        loss_sum = sup_loss + mixup_weight * unsup_loss
        return loss_sum

    def update_soft(self, data, labels, train_index):
        out = self.forward(data.x, data.edge_index)
        out = F.log_softmax(out, dim=-1)
        loss_sum = F.nll_loss(out[train_index], labels[train_index])
        return loss_sum

    def loss(self, data, opt):
        device = data.x.device
        train_index = torch.where(data.train_mask)[0].to(device)
        rand_n = random.randint(0, 1)
        if rand_n == 0:
            vector_labels = get_one_hot_label(data.y, train_index).to(device)
            return self.update_mix(data, vector_labels, train_index, opt)
        else:
            return self.update_soft(data, data.y, train_index)

    def predict(self, data):
        prediction = self.forward(data.x, data.edge_index, mix_hidden=False)
        prediction = F.log_softmax(prediction, dim=-1)
        return prediction

    def predict_noise(self, data, tau=1):
        out = self.forward(data.x, data.edge_index, mix_hidden=False) / tau
        return F.softmax(out, dim=-1)


@register_model("gcnmix")
class GCNMix(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--k", type=int, default=10)
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--rampup-starts", type=int, default=300)
        parser.add_argument("--rampup_ends", type=int, default=1000)
        parser.add_argument("--mixup-consistency", type=float, default=5.0)
        parser.add_argument("--ema-decay", type=float, default=0.999)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(in_feat=args.num_features,
                   hidden_size=args.hidden_size,
                   num_classes=args.num_classes,
                   k=args.k,
                   temperature=args.temperature,
                   alpha=args.alpha,
                   rampup_starts=args.rampup_starts,
                   rampup_ends=args.rampup_ends,
                   final_consistency_weight=args.mixup_consistency,
                   ema_decay=args.ema_decay,
                   dropout=args.dropout)

    def __init__(self, in_feat, hidden_size, num_classes, k, temperature, alpha,
                 rampup_starts, rampup_ends, final_consistency_weight, ema_decay, dropout):
        super(GCNMix, self).__init__()
        self.final_consistency_weight = final_consistency_weight
        self.rampup_starts = rampup_starts
        self.rampup_ends = rampup_ends
        self.ema_decay = ema_decay

        self.base_gnn = BaseGNNMix(in_feat, hidden_size, num_classes, k, temperature, alpha, dropout)
        self.ema_gnn = BaseGNNMix(in_feat, hidden_size, num_classes, k, temperature, alpha, dropout)
        for param in self.ema_gnn.parameters():
            param.detach_()

        self.epoch = 0

    def forward(self):
        pass

    def loss(self, data):
        opt = {
            "epoch": self.epoch,
            "final_consistency_weight": self.final_consistency_weight,
            "rampup_starts": self.rampup_starts,
            "rampup_ends": self.rampup_ends
        }
        self.base_gnn.train()
        loss_n = self.base_gnn.loss(data, opt)

        alpha = min(1 - 1/(self.epoch+1), self.ema_decay)
        for ema_param, param in zip(self.ema_gnn.parameters(), self.base_gnn.parameters()):
            ema_param.data.mul_(alpha).add_((1-alpha) * param.data)
        self.epoch += 1
        return loss_n

    def predict(self, data):
        return self.ema_gnn.predict(data)
