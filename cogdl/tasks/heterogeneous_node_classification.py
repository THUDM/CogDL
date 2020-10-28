import copy
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.models.supervised_model import SupervisedHeterogeneousNodeClassificationModel
from cogdl.trainers.supervised_trainer import (
    SupervisedHeterogeneousNodeClassificationTrainer,
    SupervisedHomogeneousNodeClassificationTrainer,
)
from . import BaseTask, register_task


@register_task("heterogeneous_node_classification")
class HeterogeneousNodeClassification(BaseTask):
    """Heterogeneous Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--num-features", type=int)
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(HeterogeneousNodeClassification, self).__init__(args)

        self.device = args.device_id[0] if not args.cpu else "cpu"
        dataset = build_dataset(args) if dataset is None else dataset

        if not args.cpu:
            dataset.apply_to_device(self.device)
        self.dataset = dataset
        self.data = dataset.data

        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.num_edge = dataset.num_edge
        args.num_nodes = dataset.num_nodes

        model = build_model(args) if model is None else model
        self.model: SupervisedHeterogeneousNodeClassificationModel = model.to(
            self.device
        )

        self.trainer: Optional[
            SupervisedHeterogeneousNodeClassificationTrainer
        ] = self.model.get_trainer(HeterogeneousNodeClassification, args)(
            self.args
        ) if self.model.get_trainer(
            HeterogeneousNodeClassification, args
        ) else None

        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def train(self):
        if self.trainer:
            self.trainer.fit(self.model, self.dataset)
        else:
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
                    if val_acc >= best_score:
                        best_loss = val_loss
                        best_score = val_acc
                        best_model = copy.deepcopy(self.model.state_dict())
                    min_loss = np.min((min_loss, val_loss))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        self.model.load_state_dict(best_model)
                        epoch_iter.close()
                        break
        test_f1, _ = self._test_step(split="test")
        print(f"Test f1 = {test_f1}")
        return dict(f1=test_f1)

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        if split == "train":
            loss, f1 = self.model.evaluate(
                self.data, self.data.train_node, self.data.train_target
            )
        elif split == "val":
            loss, f1 = self.model.evaluate(
                self.data, self.data.valid_node, self.data.valid_target
            )
        else:
            loss, f1 = self.model.evaluate(
                self.data, self.data.test_node, self.data.test_target
            )
        return f1, loss
