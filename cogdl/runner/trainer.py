from typing import Optional
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from cogdl.wrappers.data_wrapper.base_data_wrapper import DataWrapper
from cogdl.wrappers.model_wrapper.base_model_wrapper import ModelWrapper, EmbeddingModelWrapper
from cogdl.runner.trainer_utils import evaluation_comp
from cogdl.runner.embed_trainer import EmbeddingTrainer
from cogdl.data import Graph


def move_to_device(batch, device):
    if isinstance(batch, list) or isinstance(batch, tuple):
        if isinstance(batch, tuple):
            batch = list(batch)
        for i, x in enumerate(batch):
            if torch.is_tensor(x):
                batch[i] = x.to(device)
            elif isinstance(x, Graph):
                x.to(device)
    elif torch.is_tensor(batch) or isinstance(batch, Graph):
        batch = batch.to(device)
    return batch


class Trainer(object):
    def __init__(
        self,
        max_epoch: int,
        nstage: int = 1,
        cpu: bool = False,
        device_ids: Optional[list] = None,
        distributed_training: bool = False,
        distributed_inference: bool = False,
        hooks: Optional[list] = None,
        monitor: str = "val_acc",
        early_stopping: bool = True,
        patience: int = 100,
        eval_step: int = 1,
        save_embedding_path: Optional[str] = None,
        cpu_inference: bool = False,
    ):
        self.max_epoch = max_epoch
        self.nstage = nstage
        self.patience = patience
        self.early_stopping = early_stopping
        self.eval_step = eval_step
        self.monitor = monitor

        self.cpu = cpu
        self.devices, self.world_size = self.set_device(device_ids)

        self.distributed_training = distributed_training
        self.distributed_inference = distributed_inference

        self.cpu_inference = cpu_inference

        self.on_train_batch_transform = None
        self.on_eval_batch_transform = None

        self.save_embedding_path = save_embedding_path

        if hooks is None:
            self.custom_hooks = []
        else:
            self.custom_hooks = hooks

    def set_device(self, device_ids: Optional[list]):
        """
        Return: devices, world_size
        """
        if device_ids is None or self.cpu:
            return [torch.device("cpu")], 0

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

    def prepare_data_wrapper(self, dataset_w, rank):
        dataset_w.pre_transform()
        dataset_w.prepare_training_data(self.distributed_training, rank, self.world_size)
        dataset_w.prepare_val_data(False, rank, self.world_size)
        dataset_w.prepare_test_data(False, rank, self.world_size)

    def build_optimizer(self, model_w):
        opt_wrap = model_w.setup_optimizer()
        if isinstance(opt_wrap, list) or isinstance(opt_wrap, tuple):
            assert len(opt_wrap) == 2
            optimizers, lr_schedulars = opt_wrap
        else:
            optimizers = opt_wrap
            lr_schedulars = None

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if lr_schedulars and not isinstance(lr_schedulars, list):
            lr_schedulars = [lr_schedulars]
        return optimizers, lr_schedulars

    def train(self, model_w, dataset_w, rank):
        self.prepare_data_wrapper(dataset_w, rank)

        model_w.to(rank)

        optimizers, lr_schedulars = self.build_optimizer(model_w)
        best_index, compare_fn = evaluation_comp(self.monitor)
        best_model_w = None

        patience = 0
        for stage in range(self.nstage):
            with torch.no_grad():
                pre_stage_out = model_w.pre_stage(stage, dataset_w)
                dataset_w.pre_stage(stage, pre_stage_out)

            epoch_iter = tqdm(range(self.max_epoch))
            for epoch in epoch_iter:
                # inductive setting ..
                dataset_w.train()
                train_loader = dataset_w.on_training_wrapper()
                training_loss = self.training_step(model_w, train_loader, optimizers, lr_schedulars, rank)

                val_loader = dataset_w.on_val_wrapper()
                if val_loader is not None and (epoch % self.eval_step) == 0:
                    # inductive setting ..
                    dataset_w.eval()
                    # do validation in inference device
                    val_result = self.val_step(model_w, val_loader, rank)
                    if val_result is None:
                        epoch_iter.set_description(f"Epoch {epoch}, TrainLoss: {training_loss: .4f}")
                        continue

                    monitoring = val_result[self.monitor]
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

            with torch.no_grad():
                post_stage_out = model_w.post_stage(stage, dataset_w)
                dataset_w.post_stage(stage, post_stage_out)

            if best_model_w is None:
                best_model_w = model_w
        return best_model_w

    def validate(self, model_w, val_loader, device):
        if self.cpu_inference:
            model_w.to("cpu")
            _device = "cpu"
        else:
            _device = device
        result = self.val_step(model_w, val_loader, _device)
        model_w.to(device)
        return result

    def training_step(self, model_w, train_loader, optimizers, lr_schedulars, device):
        model_w.train()
        losses = []
        for batch in train_loader:
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            loss = model_w.on_train_step(batch)

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_w.parameters(), 5)

            for optimizer in optimizers:
                optimizer.step()

            losses.append(loss.item())
        if lr_schedulars is not None:
            for lr_schedular in lr_schedulars:
                lr_schedular.step()
        return np.mean(losses)

    def test(self, model_w: ModelWrapper, dataset_w: DataWrapper, device):
        model_w.eval()
        if self.cpu_inference:
            model_w.to("cpu")
            _device = device
        else:
            _device = device

        test_loader = dataset_w.on_test_wrapper()
        result = self.test_step(model_w, test_loader, _device)

        model_w.on_evaluate(dataset_w.evaluation_wrapper())
        result_eval = model_w.collect_notes()

        model_w.to(device)
        if result is not None:
            return result
        else:
            return result_eval

    @torch.no_grad()
    def val_step(self, model_w, val_loader, device):
        model_w.eval()
        for batch in val_loader:
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            model_w.on_val_step(batch)
        return model_w.collect_notes()

    @torch.no_grad()
    def test_step(self, model_w, test_loader, device):
        model_w.eval()
        for batch in test_loader:
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            model_w.on_test_step(batch)
        return model_w.collect_notes()

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
