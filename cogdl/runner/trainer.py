from typing import Optional
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from cogdl.wrappers.data_wrapper.base_data_wrapper import DataWrapper
from cogdl.wrappers.model_wrapper.base_model_wrapper import ModelWrapper, EmbeddingModelWrapper
from cogdl.runner.trainer_utils import merge_batch_indexes, evaluation_comp
from cogdl.runner.embed_trainer import EmbeddingTrainer
from cogdl.data import Graph


def move_to_device(batch, device):
    if isinstance(batch, list) or isinstance(batch, tuple):
        if isinstance(batch, tuple):
            batch = list(batch)
        for i, x in enumerate(batch):
            if torch.is_tensor(x) or isinstance(x, Graph):
                x.to(device)
    elif torch.is_tensor(batch) or isinstance(batch, Graph):
        batch.to(device)
    elif hasattr(batch, "apply_to_device"):
        batch.apply_to_device(device)
    return batch


class Trainer(object):
    def __init__(
        self,
        max_epoch: int,
        device_ids: list,
        distributed_training: bool = False,
        distributed_inference: bool = False,
        hooks: Optional[list] = None,
        monitor: str = "val_acc",
        early_stopping: bool = True,
        patience: int = 100,
        eval_step: int = 1,
        eval_model_cpu: bool = False,
        eval_data_cpu: bool = False,
        save_embedding_path: Optional[str] = None,
    ):
        self.max_epoch = max_epoch
        self.patience = patience
        self.early_stopping = early_stopping
        self.eval_step = eval_step
        self.monitor = monitor

        self.devices, self.world_size = self.set_device(device_ids)

        self.distributed_training = distributed_training
        self.distributed_inference = distributed_inference

        self.eval_model_cpu = eval_model_cpu
        self.eval_data_cpu = eval_data_cpu
        self.device_cpu = "cpu"

        self.on_train_batch_transform = None
        self.on_eval_batch_transform = None

        self.save_embedding_path = save_embedding_path

        if hooks is None:
            self.custom_hooks = []
        else:
            self.custom_hooks = hooks

    @staticmethod
    def set_device(device_ids: list):
        """
        Return: devices, world_size
        """
        if isinstance(device_ids, int) and device_ids > 0:
            device_ids = [device_ids]
        elif isinstance(device_ids, list):
            pass
        else:
            raise ValueError("`device_id` has to be list of integers")
        if len(device_ids) == 0:
            return torch.device("cpu"), 0
        else:
            return [torch.device(i) for i in device_ids], len(device_ids)

    def run(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        # set default loss_fn and evaluator for model_wrapper
        # mainly for in-cogdl setting

        # for network/graph embedding models
        if isinstance(model_w, EmbeddingModelWrapper):
            return EmbeddingTrainer(self.save_embedding_path).run(model_w, dataset_w)

        # for deep learning models
        model_w.default_loss_fn = dataset_w.get_default_loss_fn()
        model_w.default_evaluator = dataset_w.get_default_evaluator()
        if self.distributed_training and self.world_size > 1:
            best_model_w = self.dist_train(model_w, dataset_w)
        else:
            best_model_w = self.train(model_w, dataset_w, self.devices[0])
        final = self.test(best_model_w, dataset_w, self.devices[0])
        return final

    def dist_train(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        mp.set_start_method("spawn")

        device_count = torch.cuda.device_count()
        if device_count < self.world_size:
            size = device_count
            print(f"Available device count ({device_count}) is less than world size ({self.world_size})")
        else:
            size = self.world_size

        print(f"Let's using {size} GPUs.")

        processes = []
        for rank in range(size):
            p = mp.Process(target=self.train, args=(model_w, dataset_w, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # model.load_state_dict(torch.load(os.path.join("./checkpoints", f"{self.args.model}_{self.args.dataset}.pt")))
        # self.model = model
        # self.data = dataset[0]
        # self.dataset = dataset
        return model_w

    def train(self, model_w, dataset_w, rank):
        self.on_train_batch_transform = dataset_w.on_transform
        self.on_eval_batch_transform = dataset_w.on_transform
        dataset_w.pre_transform()

        model_w.to(rank)
        optimizer = model_w.setup_optimizer()
        train_loader = dataset_w.on_training_wrapper()
        val_loader = dataset_w.on_val_wrapper()
        # test_loader = dataset_w.on_test_wrapper()

        best_index, compare_fn = evaluation_comp(self.monitor)
        best_model_w = None

        patience = 0
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            # inductive setting ..
            dataset_w.train()
            training_loss = self.training_step(model_w, train_loader, optimizer, rank)
            if val_loader is not None and (epoch % self.eval_step) == 0:
                # inductive setting ..
                dataset_w.eval()
                monitoring = self.validate(model_w, val_loader, rank)
                if compare_fn(monitoring, best_index):
                    best_index = monitoring
                    patience = 0
                    best_model_w = model_w
                else:
                    patience += 1
                    if self.early_stopping and patience >= self.patience:
                        break
                epoch_iter.set_description(
                    f"Epoch {epoch}, TrainLoss: {training_loss: .4f}, ValMetric: {monitoring: .4f}"
                )
            else:
                epoch_iter.set_description(f"Epoch {epoch}, TrainLoss: {training_loss: .4f}")

        if best_model_w is None:
            best_model_w = model_w
        return best_model_w

    @torch.no_grad()
    def validate(self, model_w, val_loader, rank):
        result = self.val_step(model_w, val_loader, rank)
        return result[self.monitor]

    def training_step(self, model_w, train_loader, optimizer, device):
        model_w.train()
        losses = []
        for batch in train_loader:
            if self.on_train_batch_transform is not None:
                batch = self.on_train_batch_transform(batch)
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            loss = model_w.train_step(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def test(self, model_w: ModelWrapper, dataset_w: DataWrapper, device):
        test_loader = dataset_w.on_test_wrapper()
        result = self.test_step(model_w, test_loader, device)
        return result

    @torch.no_grad()
    def val_step(self, model_w, val_loader, device):
        outputs = []
        model_w.eval()
        for batch in val_loader:
            if self.on_eval_batch_transform is not None:
                batch = self.on_eval_batch_transform(batch)
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            out = model_w.val_step(batch)
            outputs.append(out)
        return merge_batch_indexes(outputs)

    @torch.no_grad()
    def test_step(self, model_w, test_loader, device):
        outputs = []
        model_w.eval()
        for batch in test_loader:
            if self.on_eval_batch_transform is not None:
                batch = self.on_eval_batch_transform(batch)
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            out = model_w.test_step(batch)
            outputs.append(out)
        return merge_batch_indexes(outputs)

    def distributed_dataloader_proc(self, dataset_w: DataWrapper, rank):
        # TODO: just a toy implementation
        train_loader = dataset_w.on_training_wrapper()
        dataset = train_loader.dataset
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        train_loader.sampler = sampler

    def distributed_model_proc(self, model_w: ModelWrapper, rank):
        _model = model_w.wrapped_model
        ddp_model = DistributedDataParallel(_model, device_ids=[rank])
        # _model = ddp_model.module
        model_w.wrapped_model = ddp_model
