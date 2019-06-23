import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cognitive_graph import options
from cognitive_graph.datasets import build_dataset
from cognitive_graph.models import build_model

from . import BaseTask, register_task


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--num-features", type=int)
        # fmt: on

    def __init__(self, args):
        super(NodeClassification, self).__init__(args)

        dataset = build_dataset(args)
        data = dataset[0]
        self.data = data.cuda()
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        model = build_model(args)
        self.model = model.cuda()
        self.patience = args.patience

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def train(self, num_epoch):
        epoch_iter = tqdm(range(num_epoch))
        patience = 0
        best_score = 0
        for epoch in epoch_iter:
            self._train_step()
            train_acc = self._test_step(split="train")
            val_acc = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            )
            if val_acc > best_score:
                best_score = val_acc
                best_model = copy.deepcopy(self.model)
                patience = 0
            else:
                patience += 1
                if patience > self.patience:
                    self.model = best_model
                    epoch_iter.close()
                    break
        test_acc = self._test_step(split="test")
        print(f"Test accuracy = {test_acc}")
        return test_acc

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        F.nll_loss(
            self.model(self.data.x, self.data.edge_index)[self.data.train_mask],
            self.data.y[self.data.train_mask],
        ).backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        _, mask = list(self.data(f"{split}_mask"))[0]
        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc
