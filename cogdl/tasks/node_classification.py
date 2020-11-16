import copy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.models.supervised_model import SupervisedHomogeneousNodeClassificationModel
from cogdl.trainers.supervised_trainer import (
    SupervisedHomogeneousNodeClassificationTrainer,
)
from cogdl.trainers.sampled_trainer import SampledTrainer

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

    def __init__(
        self,
        args,
        dataset=None,
        model: Optional[SupervisedHomogeneousNodeClassificationModel] = None,
    ):
        super(NodeClassification, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device_id[0] if not args.cpu else "cpu"
        dataset = build_dataset(args) if dataset is None else dataset

        self.dataset = dataset
        self.data = dataset[0]
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.num_nodes = dataset.data.x.shape[0]

        self.model: SupervisedHomogeneousNodeClassificationModel = build_model(args) if model is None else model

        self.trainer: Optional[
            SupervisedHomogeneousNodeClassificationTrainer
        ] = self.model.get_trainer(NodeClassification, self.args)(
            self.args
        ) if self.model.get_trainer(
            NodeClassification, self.args
        ) else None

        if not self.trainer:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            ) if not hasattr(self.model, "get_optimizer") else self.model.get_optimizer(args)
            self.data.apply(lambda x: x.to(self.device))
            self.model: SupervisedHomogeneousNodeClassificationModel = self.model.to(
                self.device
            )
            self.patience = args.patience
            self.max_epoch = args.max_epoch

    def train(self):
        if self.trainer:
            # if issubclass(type(self.trainer), SampledTrainer):
            #     self.model = self.trainer.fit(self.model, self.dataset)
            # else:
            #     return dict(Acc=self.trainer.fit(self.model, self.dataset.data))
            result = self.trainer.fit(self.model, self.dataset)
            if isinstance(result, torch.nn.Module):
                self.model = result
            else:
                return result
        else:
            epoch_iter = tqdm(range(self.max_epoch))
            patience = 0
            best_score = 0
            best_loss = np.inf
            max_score = 0
            min_loss = np.inf
            best_model = copy.deepcopy(self.model)
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
                        epoch_iter.close()
                        break
            print(f"Valid accurracy = {best_score}")
            self.model = best_model
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_acc}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val", logits=None):
        self.model.eval()
        logits = logits if logits else self.model.predict(self.data)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        loss = F.nll_loss(logits[mask], self.data.y[mask]).item()

        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss
