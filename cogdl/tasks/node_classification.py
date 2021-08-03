import argparse
import copy

import numpy as np
import torch
from tqdm import tqdm

from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--missing-rate", type=int, default=0, help="missing rate, from 0 to 100")
        parser.add_argument("--inference", action="store_true")
        # fmt: on

    def __init__(
        self,
        args,
        dataset=None,
        model=None,
    ):
        super(NodeClassification, self).__init__(args)

        self.args = args
        self.infer = hasattr(args, "inference") and args.inference

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        dataset = build_dataset(args) if dataset is None else dataset

        self.dataset = dataset
        self.data = dataset[0]
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        args.num_nodes = dataset.data.x.shape[0]
        args.edge_attr_size = dataset.edge_attr_size

        if args.actnn:
            try:
                from actnn import config
            except Exception:
                print("Please install the actnn library first.")
                exit(1)
            config.group_size = 256 if args.hidden_size >= 256 else 64
            config.adaptive_conv_scheme = False
            config.adaptive_bn_scheme = False

        self.model = build_model(args) if model is None else model
        self.model.set_device(self.device)
        self.model_name = self.model.__class__.__name__

        self.set_loss_fn(dataset)
        self.set_evaluator(dataset)

        self.trainer = self.get_trainer(self.args)
        if not self.trainer:
            self.optimizer = (
                torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                if not hasattr(self.model, "get_optimizer")
                else self.model.get_optimizer(args)
            )
            self.data.apply(lambda x: x.to(self.device))
            self.model = self.model.to(self.device)
            self.patience = args.patience
            self.max_epoch = args.max_epoch

    def preprocess(self):
        self.data.add_remaining_self_loops()

    def train(self):
        if self.infer:
            self.preprocess()
            self.inference()
        elif self.trainer:
            result = self.trainer.fit(self.model, self.dataset)
            if issubclass(type(result), torch.nn.Module):
                self.model = result
                self.model.to(self.data.x.device)
            else:
                return result
        else:
            self.preprocess()
            epoch_iter = tqdm(range(self.max_epoch))
            patience = 0
            best_score = 0
            # best_loss = np.inf
            max_score = 0
            min_loss = np.inf
            best_model = copy.deepcopy(self.model)

            for epoch in epoch_iter:
                self._train_step()
                acc, losses = self._test_step()
                train_acc = acc["train"]
                val_acc = acc["val"]
                val_loss = losses["val"]
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, ValLoss:{val_loss: .4f}"
                )
                # if val_loss <= min_loss or val_acc >= max_score:
                #     if val_loss <= best_loss:  # and val_acc >= best_score:
                if val_loss <= min_loss or val_acc >= best_score:
                    if val_acc >= best_score:
                        # best_loss = val_loss
                        best_score = val_acc
                        best_model = copy.deepcopy(self.model)
                    min_loss = np.min((min_loss, val_loss.cpu()))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
            print(f"Valid accurracy = {best_score: .4f}")
            self.model = best_model
        acc, _ = self._test_step(post=True)
        val_acc, test_acc = acc["val"], acc["test"]

        print(f"Test accuracy = {test_acc:.4f}")

        return dict(Acc=test_acc, ValAcc=val_acc)

    def _train_step(self):
        self.data.train()
        self.model.train()
        self.optimizer.zero_grad()
        self.model.node_classification_loss(self.data).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

    def _test_step(self, split=None, post=False):
        self.data.eval()
        self.model.eval()
        with torch.no_grad():
            logits = self.model.predict(self.data)
            if post and hasattr(self.model, "postprocess"):
                logits = self.model.postprocess(self.data, logits)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        elif split == "test":
            mask = self.data.test_mask
        else:
            mask = None

        if mask is not None:
            loss = self.loss_fn(logits[mask], self.data.y[mask])
            metric = self.evaluator(logits[mask], self.data.y[mask])
            return metric, loss
        else:
            masks = {x: self.data[x + "_mask"] for x in ["train", "val", "test"]}
            metrics = {key: self.evaluator(logits[mask], self.data.y[mask]) for key, mask in masks.items()}
            losses = {key: self.loss_fn(logits[mask], self.data.y[mask]) for key, mask in masks.items()}
            return metrics, losses

    def inference(self):
        self.data.eval()
        self.model.eval()
        with torch.no_grad():
            logits = self.model.predict(self.data)
        metric = self.evaluator(logits[self.data.test_mask], self.data.y[self.data.test_mask])
        print(f"Metric in test set: {metric: .4f}")
        key = f"{self.args.model}_{self.args.dataset}.pred"
        torch.save(logits, key)
        print(f"Prediction results saved in {key}")
