import argparse
import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.models.supervised_model import SupervisedHomogeneousNodeClassificationModel
from cogdl.trainers.sampled_trainer import SAINTTrainer
from cogdl.trainers.self_task_trainer import SelfTaskTrainer

from . import BaseTask, register_task


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--missing-rate", type=int, default=0, help="missing rate, from 0 to 100")
        parser.add_argument('--sampler', default='none', type=str, help='graph samplers')
        parser.add_argument('--sample-coverage', default=20, type=float, help='sample coverage ratio')
        parser.add_argument('--size-subgraph', default=1200, type=int, help='subgraph size')
        parser.add_argument('--num-walks', default=50, type=int, help='number of random walks')
        parser.add_argument('--walk-length', default=20, type=int, help='random walk length')
        parser.add_argument('--size-frontier', default=20, type=int, help='frontier size in multidimensional random walks')
        parser.add_argument('--auxiliary-task', default="none", type=str)
        parser.add_argument('--alpha', default=10, type=float)
        parser.add_argument('--label-mask', default=0, type=float)
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

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        dataset = build_dataset(args) if dataset is None else dataset

        self.dataset = dataset
        self.data = dataset[0]
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.num_nodes = dataset.data.x.shape[0]

        self.model: SupervisedHomogeneousNodeClassificationModel = build_model(args) if model is None else model
        self.model.set_device(self.device)

        self.trainer = (
            self.model.get_trainer(NodeClassification, self.args)(self.args)
            if self.model.get_trainer(NodeClassification, self.args)
            else None
        )

        if not self.trainer:
            if self.args.sampler != "none":
                self.trainer = SAINTTrainer(self.args)
            elif self.args.auxiliary_task != "none":
                self.trainer = SelfTaskTrainer(self.args)
            else:
                self.optimizer = (
                    torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    if not hasattr(self.model, "get_optimizer")
                    else self.model.get_optimizer(args)
                )
                self.data.apply(lambda x: x.to(self.device))
                self.model: SupervisedHomogeneousNodeClassificationModel = self.model.to(self.device)
                self.patience = args.patience
                self.max_epoch = args.max_epoch

    def train(self):
        if self.trainer:
            if isinstance(self.trainer, SAINTTrainer):
                self.model = self.trainer.fit(self.model, self.dataset)
                self.data.apply(lambda x: x.to(self.device))
            else:
                result = self.trainer.fit(self.model, self.dataset)
                if issubclass(type(result), torch.nn.Module):
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
                epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
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
        self.model.node_classification_loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val", logits=None):
        self.model.eval()
        logits = logits if logits else self.model.predict(self.data)
        logits = F.log_softmax(logits, dim=-1)
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
