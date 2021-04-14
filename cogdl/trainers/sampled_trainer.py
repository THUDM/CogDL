from abc import abstractmethod
import argparse
import copy

import numpy as np
import torch
from tqdm import tqdm

from cogdl.data import Dataset
from cogdl.data.sampler import (
    NodeSampler,
    EdgeSampler,
    RWSampler,
    MRWSampler,
    NeighborSampler,
    ClusteredLoader,
)
from cogdl.models.supervised_model import SupervisedModel
from cogdl.trainers.base_trainer import BaseTrainer
from . import register_trainer


class SampledTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--eval-step", type=int, default=3)
        parser.add_argument("--batch-size", type=int, default=128)
        # fmt: on

    @abstractmethod
    def fit(self, model: SupervisedModel, dataset: Dataset):
        raise NotImplementedError

    @abstractmethod
    def _train_step(self):
        raise NotImplementedError

    @abstractmethod
    def _test_step(self, split="val"):
        raise NotImplementedError

    def __init__(self, args):
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.loss_fn, self.evaluator = None, None
        self.data, self.train_loader, self.optimizer = None, None, None
        self.eval_step = args.eval_step if hasattr(args, "eval_step") else 1
        self.num_workers = args.num_workers if hasattr(args, "num_workers") else 0
        self.batch_size = args.batch_size

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        max_score = 0
        min_loss = np.inf
        best_model = copy.deepcopy(self.model)

        for epoch in epoch_iter:
            self._train_step()
            if (epoch + 1) % self.eval_step == 0:
                acc, loss = self._test_step()
                train_acc = acc["train"]
                val_acc = acc["val"]
                val_loss = loss["val"]
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Train Acc/F1: {train_acc:.4f}, Val Acc/F1: {val_acc:.4f}"
                )
                self.model = self.model.to(self.device)
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_loss <= min_loss:
                        best_model = copy.deepcopy(self.model)
                    min_loss = np.min((min_loss, val_loss.cpu()))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
        return best_model


@register_trainer("graphsaint")
class SAINTTrainer(SampledTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        SampledTrainer.add_args(parser)
        parser.add_argument("--sampler", default="node", type=str, help="graph samplers")
        parser.add_argument("--sample-coverage", default=20, type=float, help="sample coverage ratio")
        parser.add_argument("--size-subgraph", default=1200, type=int, help="subgraph size")
        parser.add_argument("--num-walks", default=50, type=int, help="number of random walks")
        parser.add_argument("--walk-length", default=20, type=int, help="random walk length")
        parser.add_argument("--size-frontier", default=20, type=int, help="frontier size in multidimensional random walks")
        parser.add_argument("--valid-cpu", action="store_true", help="run validation on cpu")
        # fmt: on

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(SAINTTrainer, self).__init__(args)
        self.valid_cpu = args.valid_cpu
        self.args_sampler = self.sampler_from_args(args)

    def sampler_from_args(self, args):
        args_sampler = {
            "sampler": args.sampler,
            "sample_coverage": args.sample_coverage,
            "size_subgraph": args.size_subgraph,
            "num_walks": args.num_walks,
            "walk_length": args.walk_length,
            "size_frontier": args.size_frontier,
        }
        return args_sampler

    def fit(self, model: SupervisedModel, dataset: Dataset):
        self.dataset = dataset
        self.data = dataset.data
        self.model = model.to(self.device)
        self.evaluator = dataset.get_evaluator()
        self.loss_fn = dataset.get_loss_fn()

        if self.args_sampler["sampler"] == "node":
            self.sampler = NodeSampler(self.data, self.args_sampler)
        elif self.args_sampler["sampler"] == "edge":
            self.sampler = EdgeSampler(self.data, self.args_sampler)
        elif self.args_sampler["sampler"] == "rw":
            self.sampler = RWSampler(self.data, self.args_sampler)
        elif self.args_sampler["sampler"] == "mrw":
            self.sampler = MRWSampler(self.data, self.args_sampler)
        else:
            raise NotImplementedError

        # self.train_dataset = SAINTDataset(dataset, self.args_sampler)
        # self.train_loader = SAINTDataLoader(
        #     dataset=train_dataset,
        #     num_workers=self.num_workers,
        #     persistent_workers=True,
        #     pin_memory=True
        # )
        # self.set_data_model(dataset, model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return self.train()

    def _train_step(self):
        self.data = self.sampler.one_batch("train")
        self.data.to(self.device)

        self.model = self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        mask = self.data.train_mask
        if len(self.data.y.shape) > 1:
            logits = self.model.predict(self.data)
            weight = self.data.norm_loss[mask].unsqueeze(1)
            loss = torch.nn.BCEWithLogitsLoss(reduction="sum", weight=weight)(logits[mask], self.data.y[mask].float())
        else:
            logits = torch.nn.functional.log_softmax(self.model.predict(self.data))
            loss = (
                torch.nn.NLLLoss(reduction="none")(logits[mask], self.data.y[mask]) * self.data.norm_loss[mask]
            ).sum()
        loss.backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        self.data = self.sampler.one_batch(split)
        if split != "train" and self.valid_cpu:
            self.model = self.model.cpu()
        else:
            self.data.apply(lambda x: x.to(self.device))
        self.model.eval()
        masks = {"train": self.data.train_mask, "val": self.data.val_mask, "test": self.data.test_mask}
        with torch.no_grad():
            logits = self.model.predict(self.data)

        # if isinstance(self.dataset, SAINTDataset):
        #     weight = self.data.norm_loss.unsqueeze(1)
        #     loss = torch.nn.BCEWithLogitsLoss(reduction="sum", weight=weight)(logits, self.data.y.float())
        #     metric = multilabel_f1(logits[mask], self.data.y[mask])
        # else:
        #     loss = (
        #         torch.nn.NLLLoss(reduction="none")(F.log_softmax(logits[mask]), self.data.y[mask])
        #         * self.data.norm_loss[mask]
        #     ).sum()
        #     pred = logits[mask].max(1)[1]
        #     metric = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()

        loss = {key: self.loss_fn(logits[val], self.data.y[val]) for key, val in masks.items()}
        metric = {key: self.evaluator(logits[val], self.data.y[val]) for key, val in masks.items()}
        return metric, loss


@register_trainer("neighborsampler")
class NeighborSamplingTrainer(SampledTrainer):
    model: torch.nn.Module

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        SampledTrainer.add_args(parser)
        # fmt: on

    def __init__(self, args):
        super(NeighborSamplingTrainer, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.sample_size = args.sample_size

    def fit(self, model, dataset):
        self.data = dataset[0]
        self.data.add_remaining_self_loops()
        self.evaluator = dataset.get_evaluator()
        self.loss_fn = dataset.get_loss_fn()

        settings = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
        )

        if torch.__version__.split("+")[0] < "1.7.1":
            settings.pop("persistent_workers")

        self.data.train()
        self.train_loader = NeighborSampler(
            dataset=dataset,
            mask=self.data.train_mask,
            sizes=self.sample_size,
            **settings,
        )

        settings["batch_size"] *= 5
        self.data.eval()
        self.test_loader = NeighborSampler(
            dataset=dataset,
            mask=None,
            sizes=[-1],
            **settings,
        )
        self.model = model.to(self.device)
        self.model.set_data_device(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_model = self.train()
        self.model = best_model
        acc, loss = self._test_step()
        return dict(Acc=acc["test"], ValAcc=acc["val"])

    def _train_step(self):
        self.data.train()
        self.model.train()
        self.train_loader.shuffle()

        x_all = self.data.x.to(self.device)
        y_all = self.data.y.to(self.device)

        for target_id, n_id, adjs in self.train_loader:
            self.optimizer.zero_grad()
            n_id = n_id.to(x_all.device)
            target_id = target_id.to(y_all.device)
            x_src = x_all[n_id].to(self.device)

            y = y_all[target_id].to(self.device)
            loss = self.model.node_classification_loss(x_src, adjs, y)
            loss.backward()
            self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        self.data.eval()
        masks = {"train": self.data.train_mask, "val": self.data.val_mask, "test": self.data.test_mask}
        with torch.no_grad():
            logits = self.model.inference(self.data.x, self.test_loader)

        loss = {key: self.loss_fn(logits[val], self.data.y[val]) for key, val in masks.items()}
        acc = {key: self.evaluator(logits[val], self.data.y[val]) for key, val in masks.items()}
        return acc, loss

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)


@register_trainer("clustergcn")
class ClusterGCNTrainer(SampledTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        SampledTrainer.add_args(parser)
        parser.add_argument("--n-cluster", type=int, default=1000)
        parser.add_argument("--batch-size", type=int, default=20)
        # fmt: on

    def __init__(self, args):
        super(ClusterGCNTrainer, self).__init__(args)
        self.n_cluster = args.n_cluster
        self.batch_size = args.batch_size

    def fit(self, model, dataset):
        self.data = dataset[0]
        self.data.add_remaining_self_loops()
        self.model = model.to(self.device)
        self.evaluator = dataset.get_evaluator()
        self.loss_fn = dataset.get_loss_fn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        settings = dict(
            batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True
        )

        if torch.__version__.split("+")[0] < "1.7.1":
            settings.pop("persistent_workers")

        self.data.train()
        self.train_loader = ClusteredLoader(
            dataset,
            self.n_cluster,
            method="metis",
            **settings,
        )
        best_model = self.train()
        self.model = best_model
        metric, loss = self._test_step()

        return dict(Acc=metric["test"], ValAcc=metric["val"])

    def _train_step(self):
        self.model.train()
        self.data.train()
        self.train_loader.shuffle()
        total_loss = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            loss = self.model.node_classification_loss(batch)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        self.data.eval()
        data = self.data
        self.model = self.model.cpu()
        masks = {"train": self.data.train_mask, "val": self.data.val_mask, "test": self.data.test_mask}
        with torch.no_grad():
            logits = self.model(data)
        loss = {key: self.loss_fn(logits[val], self.data.y[val]) for key, val in masks.items()}
        metric = {key: self.evaluator(logits[val], self.data.y[val]) for key, val in masks.items()}
        return metric, loss


@register_trainer("random_cluster")
class RandomClusterTrainer(SampledTrainer):
    @staticmethod
    def add_args(parser):
        # fmt: off
        SampledTrainer.add_args(parser)
        parser.add_argument("--n-cluster", type=int, default=10)
        # fmt: on

    def __init__(self, args):
        super(RandomClusterTrainer, self).__init__(args)
        self.patience = args.patience // args.eval_step
        self.n_cluster = args.n_cluster
        self.eval_step = args.eval_step
        self.data, self.optimizer, self.evaluator, self.loss_fn = None, None, None, None

    def fit(self, model, dataset):
        self.model = model.to(self.device)
        self.data = dataset[0]
        self.data.add_remaining_self_loops()

        self.loss_fn = dataset.get_loss_fn()
        self.evaluator = dataset.get_evaluator()

        settings = dict(num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

        if torch.__version__.split("+")[0] < "1.7.1":
            settings.pop("persistent_workers")

        self.train_loader = ClusteredLoader(dataset=dataset, n_cluster=self.n_cluster, method="random", **settings)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_model = self.train()
        self.model = best_model
        metric, loss = self._test_step()
        return dict(Acc=metric["test"], ValAcc=metric["val"])

    def _train_step(self):
        self.model.train()
        self.data.train()
        self.train_loader.shuffle()

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            loss_n = self.model.node_classification_loss(batch)
            loss_n.backward()
            self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        self.data.eval()
        self.model = self.model.to("cpu")
        data = self.data
        self.model = self.model.cpu()
        masks = {"train": self.data.train_mask, "val": self.data.val_mask, "test": self.data.test_mask}
        with torch.no_grad():
            logits = self.model.predict(data)
        loss = {key: self.loss_fn(logits[val], self.data.y[val]) for key, val in masks.items()}
        metric = {key: self.evaluator(logits[val], self.data.y[val]) for key, val in masks.items()}
        return metric, loss
