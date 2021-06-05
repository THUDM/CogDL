import argparse
import copy
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from .sampled_trainer import (
    SampledTrainer,
    ClusterGCNTrainer,
    NeighborSamplingTrainer,
    SAINTTrainer,
)

from cogdl.data.sampler import (
    NeighborSampler,
    NeighborSamplerDataset,
    ClusteredDataset,
    ClusteredLoader,
    SAINTDataset,
)
from . import register_trainer
from cogdl.trainers.base_trainer import BaseTrainer


import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class DistributedSampledTrainer(BaseTrainer):
    def __init__(self, args):
        super(DistributedSampledTrainer, self).__init__(args)
        self.args = args
        self.num_workers = args.num_workers

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--world-size", type=int, default=1)
        parser.add_argument("--master-port", type=int, default=13425)
        parser.add_argument("--dist-inference", action="store_true")
        parser.add_argument("--eval-step", type=int, default=4)
        # fmt: on

    def dist_fit(self, model, dataset):
        mp.set_start_method("spawn")

        device_count = torch.cuda.device_count()
        if device_count < self.args.world_size:
            size = device_count
            print(f"Available device count ({device_count}) is less than world size ({self.args.world_size})")
        else:
            size = self.args.world_size

        print(f"Let's using {size} GPUs.")

        self.evaluator = dataset.get_evaluator()
        self.loss_fn = dataset.get_loss_fn()

        processes = []
        for rank in range(size):
            p = mp.Process(target=self.train, args=(model, dataset, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        model.load_state_dict(torch.load(os.path.join("./checkpoints", f"{self.args.model}_{self.args.dataset}.pt")))
        self.model = model
        self.data = dataset[0]
        self.dataset = dataset
        metric, loss = self._test_step(split="test")
        return dict(Acc=metric["test"])

    def train(self, model, dataset, rank):
        print(f"Running on rank {rank}.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.args.master_port)

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=self.args.world_size)
        torch.cuda.set_device(rank)

        self.model = copy.deepcopy(model).to(rank)
        self.ddp_model = DistributedDataParallel(self.model, device_ids=[rank])
        self.model = self.ddp_model.module

        self.data = dataset[0]
        self.device = self.rank = rank

        train_dataset, loaders = self.build_dataloader(dataset, rank)
        self.train_loader, self.val_loader, self.test_loader = loaders

        self.optimizer = self.get_optimizer(self.ddp_model, self.args)
        if rank == 0:
            epoch_iter = tqdm(range(self.args.max_epoch))
        else:
            epoch_iter = range(self.args.max_epoch)

        patience = 0
        best_val_loss = np.inf
        best_val_metric = 0
        best_model = None

        for epoch in epoch_iter:
            if train_dataset is not None and hasattr(train_dataset, "shuffle"):
                train_dataset.shuffle()
            self.train_step()
            if (epoch + 1) % self.eval_step == 0:
                val_metric, val_loss = self.test_step(split="val")
                self.ddp_model = self.ddp_model.to(self.device)
                # self.model = self.model.to(self.device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(self.model)
                    best_val_metric = val_metric
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        break
                if rank == 0:
                    epoch_iter.set_description(
                        f"Epoch: {epoch:03d}, ValLoss: {val_loss:.4f}, Acc/F1: {val_metric:.4f}, BestVal Acc/F1: {best_val_metric: .4f}"
                    )
            dist.barrier()

        if rank == 0:
            os.makedirs("./checkpoints", exist_ok=True)
            checkpoint_path = os.path.join("./checkpoints", f"{self.args.model}_{self.args.dataset}.pt")
            if best_model is not None:
                print(f"Saving model to {checkpoint_path}")
                torch.save(best_model.state_dict(), checkpoint_path)

        dist.destroy_process_group()

    def test_step(self, split="val"):
        if self.device == 0:
            metric, loss = self._test_step()
            val_loss = float(loss[split])
            val_metric = float(metric[split])
            object_list = [val_metric, val_loss]
        else:
            object_list = [None, None]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0], object_list[1]

    def train_step(self):
        self._train_step()

    def get_optimizer(self, model, args):
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def build_dataloader(self, dataset, rank):
        raise NotImplementedError


@register_trainer("dist_clustergcn")
class DistributedClusterGCNTrainer(DistributedSampledTrainer, ClusterGCNTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        DistributedSampledTrainer.add_args(parser)
        ClusterGCNTrainer.add_args(parser)

    def __init__(self, args):
        super(DistributedClusterGCNTrainer, self).__init__(args)

    def fit(self, *args, **kwargs):
        return super(DistributedClusterGCNTrainer, self).dist_fit(*args, **kwargs)

    def build_dataloader(self, dataset, rank):
        if self.device != 0:
            dist.barrier()
        data = dataset[0]
        train_dataset = ClusteredDataset(dataset, self.n_cluster, self.batch_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=self.args.world_size, rank=rank
        )

        settings = dict(
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            batch_size=self.args.batch_size,
        )

        if torch.__version__.split("+")[0] < "1.7.1":
            settings.pop("persistent_wo rkers")

        data.train()
        train_loader = ClusteredLoader(
            dataset=train_dataset, n_cluster=self.args.n_cluster, method="metis", sampler=train_sampler, **settings
        )
        if self.device == 0:
            dist.barrier()

        settings["batch_size"] *= 5
        data.eval()
        test_loader = NeighborSampler(dataset=dataset, sizes=[-1], **settings)
        val_loader = test_loader
        return train_dataset, (train_loader, val_loader, test_loader)

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)


@register_trainer("dist_neighborsampler")
class DistributedNeighborSamplerTrainer(DistributedSampledTrainer, NeighborSamplingTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        DistributedSampledTrainer.add_args(parser)
        NeighborSamplingTrainer.add_args(parser)

    def __init__(self, args):
        super(DistributedNeighborSamplerTrainer, self).__init__(args)

    def fit(self, *args, **kwargs):
        super(DistributedNeighborSamplerTrainer, self).dist_fit(*args, **kwargs)

    def build_dataloader(self, dataset, rank):
        data = dataset[0]
        train_dataset = NeighborSamplerDataset(dataset, self.sample_size, self.batch_size, self.data.train_mask)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=self.args.world_size, rank=rank
        )

        settings = dict(
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            batch_size=self.args.batch_size,
        )

        if torch.__version__.split("+")[0] < "1.7.1":
            settings.pop("persistent_workers")

        data.train()
        train_loader = NeighborSampler(dataset=train_dataset, sizes=self.sample_size, sampler=train_sampler, **settings)

        settings["batch_size"] *= 5
        data.eval()
        test_loader = NeighborSampler(dataset=dataset, sizes=[-1], **settings)
        val_loader = test_loader
        return train_dataset, (train_loader, val_loader, test_loader)

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def _test_step(self, split="val"):
        if split == "test":
            if torch.__version__.split("+")[0] < "1.7.1":
                self.test_loader = NeighborSampler(
                    dataset=self.dataset,
                    sizes=[-1],
                    batch_size=self.batch_size * 10,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                )
            else:
                self.test_loader = NeighborSampler(
                    dataset=self.dataset,
                    sizes=[-1],
                    batch_size=self.batch_size * 10,
                    num_workers=self.num_workers,
                    shuffle=False,
                    persistent_workers=True,
                    pin_memory=True,
                )
        return super(DistributedNeighborSamplerTrainer, self)._test_step()


def batcher(data):
    return data[0]


@register_trainer("dist_saint")
class DistributedSAINTTrainer(DistributedSampledTrainer, SAINTTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        DistributedSampledTrainer.add_args(parser)
        SAINTTrainer.add_args(parser)

    def __init__(self, args):
        super(DistributedSAINTTrainer, self).__init__(args)

    def build_dataloader(self, dataset, rank):
        train_dataset = SAINTDataset(dataset, self.sampler_from_args(self.args))
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=self.args.world_size, rank=rank
        )

        settings = dict(
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            batch_size=self.args.batch_size,
        )

        if torch.__version__.split("+")[0] < "1.7.1":
            settings.pop("persistent_workers")

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=batcher,
        )

        test_loader = NeighborSampler(
            dataset=dataset,
            sizes=[-1],
            **settings,
        )
        val_loader = test_loader
        return train_dataset, (train_loader, val_loader, test_loader)

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def fit(self, *args, **kwargs):
        super(DistributedSAINTTrainer, self).dist_fit(*args, **kwargs)

    def _train_step(self):
        self.data.train()
        self.model.train()
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            self.model.node_classification_loss(batch).backward()
            self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        self.data.eval()
        data = self.data
        model = self.model.cpu()
        masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
        with torch.no_grad():
            logits = model.predict(data)
        loss = {key: self.loss_fn(logits[val], data.y[val]) for key, val in masks.items()}
        metric = {key: self.evaluator(logits[val], data.y[val]) for key, val in masks.items()}
        return metric, loss
