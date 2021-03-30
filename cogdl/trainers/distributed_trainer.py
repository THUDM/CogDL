import argparse
import copy
import os

import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from cogdl.data.sampler import ClusteredDataset, SAINTDataset
from cogdl.trainers.base_trainer import BaseTrainer
from . import register_trainer


def train_step(model, data_loader, optimizer, device):
    model.train()
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        model.module.node_classification_loss(batch).backward()
        optimizer.step()


def test_step(model, data, evaluator, loss_fn):
    model.eval()
    model = model.cpu()
    masks = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    with torch.no_grad():
        logits = model.predict(data)
    loss = {key: loss_fn(logits[val], data.y[val]) for key, val in masks.items()}
    metric = {key: evaluator(logits[val], data.y[val]) for key, val in masks.items()}
    return metric, loss


def batcher_clustergcn(data):
    return data[0]


def batcher_saint(data):
    return data[0]


def sampler_from_args(args):
    args_sampler = {
        "sampler": args.sampler,
        "sample_coverage": args.sample_coverage,
        "size_subgraph": args.size_subgraph,
        "num_walks": args.num_walks,
        "walk_length": args.walk_length,
        "size_frontier": args.size_frontier,
    }
    return args_sampler


def get_train_loader(dataset, args, rank):
    if args.sampler == "clustergcn":
        train_dataset = ClusteredDataset(dataset, args.n_cluster, args.batch_size, log=(rank == 0))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            # pin_memory=True,
            sampler=train_sampler,
            persistent_workers=True,
            collate_fn=batcher_clustergcn,
        )
    elif args.sampler in ["node", "edge", "rw", "mrw"]:
        train_dataset = SAINTDataset(dataset, sampler_from_args(args), log=(rank == 0))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=batcher_saint,
        )
    else:
        raise NotImplementedError(f"{args.trainer} is not implemented.")

    return train_dataset, train_loader


def train(model, dataset, args, rank, evaluator, loss_fn):
    print(f"Running on rank {rank}.")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.master_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    model = copy.deepcopy(model).to(rank)
    model = DDP(model, device_ids=[rank])

    data = dataset[0]

    train_dataset, train_loader = get_train_loader(dataset, args, rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_iter = tqdm(range(args.max_epoch)) if rank == 0 else range(args.max_epoch)
    patience = 0
    max_score = 0
    min_loss = np.inf
    best_model = None
    for epoch in epoch_iter:
        train_dataset.shuffle()
        train_step(model, train_loader, optimizer, rank)
        if (epoch + 1) % args.eval_step == 0:
            if rank == 0:
                acc, loss = test_step(model.module, data, evaluator, loss_fn)
                train_acc = acc["train"]
                val_acc = acc["val"]
                val_loss = loss["val"]
                epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                model = model.to(rank)
                object_list = [val_loss, val_acc]
            else:
                object_list = [None, None]
            dist.broadcast_object_list(object_list, src=0)
            val_loss, val_acc = object_list

            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= min_loss:
                    best_model = copy.deepcopy(model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                patience += 1
                if patience == args.patience:
                    break
        dist.barrier()

    if rank == 0:
        os.makedirs("./checkpoints", exist_ok=True)
        checkpoint_path = os.path.join("./checkpoints", f"{args.model}_{args.dataset}.pt")
        if best_model is not None:
            print(f"Saving model to {checkpoint_path}")
            torch.save(best_model.module.state_dict(), checkpoint_path)

    dist.barrier()

    dist.destroy_process_group()


@register_trainer("distributed_trainer")
class DistributedClusterGCNTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--n-cluster", type=int, default=1000)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--eval-step", type=int, default=10)
        parser.add_argument("--world-size", type=int, default=2)
        parser.add_argument("--sampler", type=str, default="clustergcn")
        parser.add_argument('--sample-coverage', default=20, type=float, help='sample coverage ratio')
        parser.add_argument('--size-subgraph', default=1200, type=int, help='subgraph size')
        parser.add_argument('--num-walks', default=50, type=int, help='number of random walks')
        parser.add_argument('--walk-length', default=20, type=int, help='random walk length')
        parser.add_argument('--size-frontier', default=20, type=int, help='frontier size in multidimensional random walks')
        parser.add_argument("--master-port", type=int, default=13579)
        # fmt: on

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        self.args = args

    def fit(self, model, dataset):
        mp.set_start_method("spawn", force=True)

        data = dataset[0]
        model = model.cpu()

        evaluator = dataset.get_evaluator()
        loss_fn = dataset.get_loss_fn()

        device_count = torch.cuda.device_count()
        if device_count < self.args.world_size:
            size = device_count
            print(f"Available device count ({device_count}) is less than world size ({self.args.world_size})")
        else:
            size = self.args.world_size

        processes = []
        for rank in range(size):
            p = Process(target=train, args=(model, dataset, self.args, rank, evaluator, loss_fn))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        model.load_state_dict(torch.load(os.path.join("./checkpoints", f"{self.args.model}_{self.args.dataset}.pt")))
        metric, loss = test_step(model, data, evaluator, loss_fn)

        return dict(Acc=metric["test"], ValAcc=metric["val"])
