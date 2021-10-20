import copy
import random
import numpy as np
import torch

from .. import ModelWrapper


class GCNMixModelWrapper(ModelWrapper):
    """
    GCNMixModelWrapper calls `forward_aux` in model
    `forward_aux` is similar to `forward` but ignores `spmm` operation.
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--rampup-starts", type=int, default=500)
        parser.add_argument("--rampup-ends", type=int, default=1000)
        parser.add_argument("--mixup-consistency", type=float, default=10.0)
        parser.add_argument("--ema-decay", type=float, default=0.999)
        parser.add_argument("--tau", type=float, default=1.0)
        parser.add_argument("--k", type=int, default=10)
        # fmt: on

    def __init__(
        self, model, optimizer_cfg, temperature, rampup_starts, rampup_ends, mixup_consistency, ema_decay, tau, k
    ):
        super(GCNMixModelWrapper, self).__init__()
        self.optimizer_cfg = optimizer_cfg
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.tau = tau
        self.k = k

        self.model = model
        self.model_ema = copy.deepcopy(self.model)

        for p in self.model_ema.parameters():
            p.detach_()
        self.epoch = 0
        self.opt = {
            "epoch": 0,
            "final_consistency_weight": mixup_consistency,
            "rampup_starts": rampup_starts,
            "rampup_ends": rampup_ends,
        }
        self.mix_loss = torch.nn.BCELoss()
        self.mix_transform = None

    def train_step(self, subgraph):
        if self.mix_transform is None:
            if len(subgraph.y.shape) > 1:
                self.mix_transform = torch.nn.Sigmoid()
            else:
                self.mix_transform = torch.nn.Softmax(-1)
        graph = subgraph
        device = graph.x.device
        train_mask = graph.train_mask

        self.opt["epoch"] += 1

        rand_n = random.randint(0, 1)
        if rand_n == 0:
            vector_labels = get_one_hot_label(graph.y, train_mask).to(device)
            loss = self.update_aux(graph, vector_labels, train_mask)
        else:
            loss = self.update_soft(graph)

        alpha = min(1 - 1 / (self.epoch + 1), self.ema_decay)
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

        return loss

    def val_step(self, subgraph):
        graph = subgraph
        val_mask = graph.val_mask
        pred = self.model_ema(graph)
        loss = self.default_loss_fn(pred[val_mask], graph.y[val_mask])

        metric = self.evaluate(pred[val_mask], graph.y[val_mask], metric="auto")

        self.note("val_loss", loss.item())
        self.note("val_metric", metric)

    def test_step(self, subgraph):
        test_mask = subgraph.test_mask
        pred = self.model_ema(subgraph)
        loss = self.default_loss_fn(pred[test_mask], subgraph.y[test_mask])

        metric = self.evaluate(pred[test_mask], subgraph.y[test_mask], metric="auto")

        self.note("test_loss", loss.item())
        self.note("test_metric", metric)

    def update_soft(self, graph):
        out = self.model(graph)
        train_mask = graph.train_mask
        loss_sum = self.default_loss_fn(out[train_mask], graph.y[train_mask])
        return loss_sum

    def update_aux(self, data, vector_labels, train_index):
        device = self.device
        train_unlabelled = torch.where(~data.train_mask)[0].to(device)
        temp_labels = torch.zeros(self.k, vector_labels.shape[0], vector_labels.shape[1]).to(device)
        with torch.no_grad():
            for i in range(self.k):
                temp_labels[i, :, :] = self.model(data) / self.tau

        target_labels = temp_labels.mean(dim=0)
        target_labels = sharpen(target_labels, self.temperature)
        vector_labels[train_unlabelled] = target_labels[train_unlabelled]
        sampled_unlabelled = torch.randint(0, train_unlabelled.shape[0], size=(train_index.shape[0],))
        train_unlabelled = train_unlabelled[sampled_unlabelled]

        def get_loss(index):
            # TODO: call `forward_aux` in model
            mix_logits, target = self.model.forward_aux(data.x, vector_labels, index, mix_hidden=True)
            # temp_loss = self.loss_f(F.softmax(mix_logits[index], -1), target)
            temp_loss = self.mix_loss(self.mix_transform(mix_logits[index]), target)
            return temp_loss

        sup_loss = get_loss(train_index)
        unsup_loss = get_loss(train_unlabelled)

        mixup_weight = get_current_consistency_weight(
            self.opt["final_consistency_weight"], self.opt["rampup_starts"], self.opt["rampup_ends"], self.opt["epoch"]
        )

        loss_sum = sup_loss + mixup_weight * unsup_loss
        return loss_sum

    def setup_optimizer(self):
        lr = self.optimizer_cfg["lr"]
        wd = self.optimizer_cfg["weight_decay"]
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)


def get_one_hot_label(labels, index):
    num_classes = int(torch.max(labels) + 1)
    target = torch.zeros(labels.shape[0], num_classes).to(labels.device)

    target[index, labels[index]] = 1
    return target


def sharpen(prob, temperature):
    prob = torch.pow(prob, 1.0 / temperature)
    row_sum = torch.sum(prob, dim=1).reshape(-1, 1)
    return prob / row_sum


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
