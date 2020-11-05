import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task

def get_batches(train_nodes, train_labels, batch_size=64, shuffle=True):
    if shuffle:
        random.shuffle(train_nodes)
    total = train_nodes.shape[0]
    for i in range(0, total, batch_size):
        if i + batch_size <= total:
            cur_nodes = train_nodes[i: i+batch_size]
            cur_labels = train_labels[cur_nodes]
            yield cur_nodes, cur_labels

@register_task("node_classification_sampling")
class NodeClassificationSampling(BaseTask):
    """Node classification task with sampling."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--batch-size", type=int, default=20)
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(NodeClassificationSampling, self).__init__(args)

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        dataset = build_dataset(args) if dataset is None else dataset
        self.data = dataset.data
        self.data.apply(lambda x: x.to(self.device))
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        model = build_model(args) if model is None else model
        self.num_nodes = self.data.x.shape[0]
        self.model = model.to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.batch_size = args.batch_size

        self.adj_list = self.data.edge_index.detach().cpu().numpy()
        self.model.set_adj(self.adj_list, self.num_nodes)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        for epoch in epoch_iter:
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            )
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                patience += 1
                if patience == self.patience:
                    self.model = best_model
                    epoch_iter.close()
                    break
        test_acc, _ = self._test_step(split="test")
        print(f"Test accuracy = {test_acc}")
        return dict(Acc=test_acc)

    def _train_step(self):
        self.model.train()
        train_nodes = np.where(self.data.train_mask.detach().cpu().numpy())[0]
        train_labels = self.data.y.detach().cpu().numpy()
        for batch_nodes, batch_labels in get_batches(train_nodes, train_labels, batch_size=self.batch_size):
            batch_nodes = torch.LongTensor(batch_nodes)
            batch_labels = torch.LongTensor(batch_labels).to(self.device)
            sampled_x, sampled_adj, var_loss = self.model.sampling(self.data.x, batch_nodes)
            self.optimizer.zero_grad()
            output = self.model(sampled_x, sampled_adj)
            loss = F.nll_loss(output, batch_labels) + 0.5 * var_loss
            loss.backward()
            self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        _, mask = list(self.data(f"{split}_mask"))[0]
        test_nodes = np.where(mask.detach().cpu().numpy())[0]
        test_labels = self.data.y.detach().cpu().numpy()
        all_loss = []
        all_acc = []
        for batch_nodes, batch_labels in get_batches(test_nodes, test_labels, batch_size=self.batch_size):
            batch_nodes = torch.LongTensor(batch_nodes)
            batch_labels = torch.LongTensor(batch_labels).to(self.device)
            sampled_x, sampled_adj, var_loss = self.model.sampling(self.data.x, batch_nodes)
            with torch.no_grad():
                logits = self.model(sampled_x, sampled_adj)
                loss = F.nll_loss(logits, batch_labels)
            pred = logits.max(1)[1]
            acc = pred.eq(self.data.y[batch_nodes]).sum().item() / batch_nodes.shape[0]

            all_loss.append(loss.item())
            all_acc.append(acc)

        return np.mean(all_acc), np.mean(all_loss)
