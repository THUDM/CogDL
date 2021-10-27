import copy
from typing import Optional
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from cogdl.wrappers.data_wrapper.base_data_wrapper import DataWrapper
from cogdl.wrappers.model_wrapper.base_model_wrapper import ModelWrapper, EmbeddingModelWrapper
from cogdl.trainer.trainer_utils import evaluation_comp, load_model, save_model, ddp_end, ddp_after_epoch, Printer
from cogdl.trainer.embed_trainer import EmbeddingTrainer
from cogdl.trainer.controller import DataController
from cogdl.loggers import build_logger
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
    elif hasattr(batch, "apply_to_device"):
        batch.apply_to_device(device)
    return batch


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None))


class Trainer(object):
    def __init__(
        self,
        max_epoch: int,
        nstage: int = 1,
        cpu: bool = False,
        checkpoint_path: str = "./checkpoints/model.pt",
        resume_training: str = False,
        device_ids: Optional[list] = None,
        distributed_training: bool = False,
        distributed_inference: bool = False,
        master_addr: str = "localhost",
        master_port: int = 10086,
        # monitor: str = "val_acc",
        early_stopping: bool = True,
        patience: int = 100,
        eval_step: int = 1,
        save_emb_path: Optional[str] = None,
        load_emb_path: Optional[str] = None,
        cpu_inference: bool = False,
        progress_bar: str = "epoch",
        clip_grad_norm: float = 5.0,
        logger: str = None,
        log_path: str = "./runs",
        project: str = "cogdl-exp",
        no_test: bool = False,
        actnn: bool = False,
        rp_ratio: int = 1,
    ):
        self.max_epoch = max_epoch
        self.nstage = nstage
        self.patience = patience
        self.early_stopping = early_stopping
        self.eval_step = eval_step
        self.monitor = None
        self.evaluation_metric = None
        self.progress_bar = progress_bar

        self.cpu = cpu
        self.devices, self.world_size = self.set_device(device_ids)
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training

        self.distributed_training = distributed_training
        self.distributed_inference = distributed_inference
        # if self.world_size <= 1:
        #     self.distributed_training = False
        #     self.distributed_inference = False
        self.master_addr = master_addr
        self.master_port = master_port

        self.cpu_inference = cpu_inference

        self.no_test = no_test

        self.on_train_batch_transform = None
        self.on_eval_batch_transform = None
        self.clip_grad_norm = clip_grad_norm

        self.save_emb_path = save_emb_path
        self.load_emb_path = load_emb_path

        self.data_controller = DataController(world_size=self.world_size, distributed=self.distributed_training)

        self.logger = build_logger(logger, log_path, project)

        self.after_epoch_hooks = []
        self.pre_epoch_hooks = []
        self.training_end_hooks = []

        if distributed_training:
            self.register_training_end_hook(ddp_end)
            self.register_out_epoch_hook(ddp_after_epoch)

        self.eval_data_back_to_cpu = False

        if actnn:
            try:
                import actnn
                from actnn.conf import config

                actnn.set_optimization_level("L3")
                if rp_ratio > 1:
                    config.group_size = 64
            except Exception:
                pass

    def register_in_epoch_hook(self, hook):
        self.pre_epoch_hooks.append(hook)

    def register_out_epoch_hook(self, hook):
        self.after_epoch_hooks.append(hook)

    def register_training_end_hook(self, hook):
        self.training_end_hooks.append(hook)

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
            return [i for i in device_ids], len(device_ids)

    def run(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        # for network/graph embedding models
        if isinstance(model_w, EmbeddingModelWrapper):
            return EmbeddingTrainer(self.save_emb_path, self.load_emb_path).run(model_w, dataset_w)

        # for deep learning models
        # set default loss_fn and evaluator for model_wrapper
        # mainly for in-cogdl setting

        model_w.default_loss_fn = dataset_w.get_default_loss_fn()
        model_w.default_evaluator = dataset_w.get_default_evaluator()
        model_w.set_evaluation_metric()

        if self.resume_training:
            model_w = load_model(model_w, self.checkpoint_path).to(self.devices[0])

        if self.distributed_training:  # and self.world_size > 1:
            torch.multiprocessing.set_sharing_strategy("file_system")
            self.dist_train(model_w, dataset_w)
        else:
            self.train(self.devices[0], model_w, dataset_w)
        best_model_w = load_model(model_w, self.checkpoint_path).to(self.devices[0])

        if self.no_test:
            return best_model_w.model

        final_test = self.evaluate(best_model_w, dataset_w)
        return final_test

    def evaluate(self, model_w: ModelWrapper, dataset_w: DataWrapper, cpu=False):
        if cpu:
            self.devices = [torch.device("cpu")]

        # disable `distributed` to inference once only
        self.distributed_training = False
        dataset_w.prepare_test_data()
        final_val = self.validate(model_w, dataset_w, self.devices[0])
        final_test = self.test(model_w, dataset_w, self.devices[0])

        if final_val is not None and "val_metric" in final_val:
            final_val[f"val_{self.evaluation_metric}"] = final_val["val_metric"]
            final_val.pop("val_metric")
            if "val_loss" in final_val:
                final_val.pop("val_loss")

        if final_test is not None and "test_metric" in final_test:
            final_test[f"test_{self.evaluation_metric}"] = final_test["test_metric"]
            final_test.pop("test_metric")
            if "test_loss" in final_test:
                final_test.pop("test_loss")

        self.logger.note(final_test)
        if final_val is not None:
            final_test.update(final_val)
        print(final_test)
        return final_test

    def dist_train(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        mp.set_start_method("spawn", force=True)

        device_count = torch.cuda.device_count()
        if device_count < self.world_size:
            size = device_count
            print(f"Available device count ({device_count}) is less than world size ({self.world_size})")
        else:
            size = self.world_size

        print(f"Let's using {size} GPUs.")

        processes = []
        for rank in range(size):
            p = mp.Process(target=self.train, args=(rank, model_w, dataset_w))

            p.start()
            print(f"Process [{rank}] starts!")
            processes.append(p)

        for p in processes:
            p.join()

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

    def initialize(self, model_w, rank=0, master_addr: str = "localhost", master_port: int = 10008):
        if self.distributed_training:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
            model_w = copy.deepcopy(model_w).to(rank)
            model_w = DistributedDataParallel(model_w, device_ids=[rank])

            module = model_w.module
            model_w, model_ddp = module, model_w
            return model_w, model_ddp
        else:
            return model_w.to(rank), None

    def train(self, rank, model_w, dataset_w):
        model_w, _ = self.initialize(model_w, rank=rank, master_addr=self.master_addr, master_port=self.master_port)
        self.data_controller.prepare_data_wrapper(dataset_w, rank)
        self.eval_data_back_to_cpu = dataset_w.data_back_to_cpu

        optimizers, lr_schedulars = self.build_optimizer(model_w)
        if optimizers[0] is None:
            return

        est = model_w.set_early_stopping()
        if isinstance(est, str):
            est_monitor = est
            best_index, compare_fn = evaluation_comp(est_monitor)
        else:
            assert len(est) == 2
            est_monitor, est_compare = est
            best_index, compare_fn = evaluation_comp(est_monitor, est_compare)
        self.monitor = est_monitor
        self.evaluation_metric = model_w.evaluation_metric

        # best_index, compare_fn = evaluation_comp(self.monitor)
        best_model_w = None

        patience = 0
        best_epoch = 0
        for stage in range(self.nstage):
            with torch.no_grad():
                pre_stage_out = model_w.pre_stage(stage, dataset_w)
                dataset_w.pre_stage(stage, pre_stage_out)
                self.data_controller.training_proc_per_stage(dataset_w, rank)

            if self.progress_bar == "epoch":
                epoch_iter = tqdm(range(self.max_epoch))
                epoch_printer = Printer(epoch_iter.set_description, rank=rank, world_size=self.world_size)
            else:
                epoch_iter = range(self.max_epoch)
                epoch_printer = Printer(print, rank=rank, world_size=self.world_size)

            self.logger.start()
            for epoch in epoch_iter:
                print_str_dict = dict()
                for hook in self.pre_epoch_hooks:
                    hook(self)

                # inductive setting ..
                dataset_w.train()
                train_loader = dataset_w.on_train_wrapper()
                training_loss = self.training_step(model_w, train_loader, optimizers, lr_schedulars, rank)

                print_str_dict["Epoch"] = epoch
                print_str_dict["train_loss"] = training_loss

                val_loader = dataset_w.on_val_wrapper()
                if val_loader is not None and (epoch % self.eval_step) == 0:
                    # inductive setting ..
                    dataset_w.eval()
                    # do validation in inference device
                    val_result = self.validate(model_w, dataset_w, rank)
                    # print(val_result)
                    if val_result is not None:
                        monitoring = val_result[self.monitor]
                        if compare_fn(monitoring, best_index):
                            best_index = monitoring
                            best_epoch = epoch
                            patience = 0
                            best_model_w = copy.deepcopy(model_w)
                        else:
                            patience += 1
                            if self.early_stopping and patience >= self.patience:
                                break
                        print_str_dict[f"val_{self.evaluation_metric}"] = monitoring

                if self.distributed_training:
                    if rank == 0:
                        epoch_printer(print_str_dict)
                        self.logger.note(print_str_dict, epoch)
                else:
                    epoch_printer(print_str_dict)
                    self.logger.note(print_str_dict, epoch)

                for hook in self.after_epoch_hooks:
                    hook(self)

            with torch.no_grad():
                model_w.eval()
                post_stage_out = model_w.post_stage(stage, dataset_w)
                dataset_w.post_stage(stage, post_stage_out)

            if best_model_w is None:
                best_model_w = copy.deepcopy(model_w)

        if self.distributed_training:
            if rank == 0:
                save_model(best_model_w.to("cpu"), self.checkpoint_path, best_epoch)
                dist.barrier()
            else:
                dist.barrier()
        else:
            save_model(best_model_w.to("cpu"), self.checkpoint_path, best_epoch)

        for hook in self.training_end_hooks:
            hook(self)

    def validate(self, model_w: ModelWrapper, dataset_w: DataWrapper, device):
        # ------- distributed training ---------
        if self.distributed_training:
            return self.distributed_test(model_w, dataset_w.on_val_wrapper(), device, self.val_step)
        # ------- distributed training ---------

        model_w.eval()
        if self.cpu_inference:
            model_w.to("cpu")
            _device = device
        else:
            _device = device

        val_loader = dataset_w.on_val_wrapper()
        result = self.val_step(model_w, val_loader, _device)

        model_w.to(device)
        return result

    def test(self, model_w: ModelWrapper, dataset_w: DataWrapper, device):
        # ------- distributed training ---------
        if self.distributed_training:
            return self.distributed_test(model_w, dataset_w.on_test_wrapper(), device, self.test_step)
        # ------- distributed training ---------

        model_w.eval()
        if self.cpu_inference:
            model_w.to("cpu")
            _device = device
        else:
            _device = device

        test_loader = dataset_w.on_test_wrapper()
        result = self.test_step(model_w, test_loader, _device)

        model_w.to(device)
        return result

    def distributed_test(self, model_w: ModelWrapper, loader, rank, fn):
        model_w.eval()
        # if rank == 0:
        if dist.get_rank() == 0:
            if self.cpu_inference:
                model_w.to("cpu")
                _device = "cpu"
            else:
                _device = rank
            result = fn(model_w, loader, _device)
            model_w.to(rank)

            object_list = [result]
        else:
            object_list = [None]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]

    def training_step(self, model_w, train_loader, optimizers, lr_schedulars, device):
        model_w.train()
        losses = []

        if self.progress_bar == "iteration":
            train_loader = tqdm(train_loader)

        for batch in train_loader:
            batch = move_to_device(batch, device)
            loss = model_w.on_train_step(batch)

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_w.parameters(), self.clip_grad_norm)

            for optimizer in optimizers:
                optimizer.step()

            losses.append(loss.item())
        if lr_schedulars is not None:
            for lr_schedular in lr_schedulars:
                lr_schedular.step()
        return np.mean(losses)

    @torch.no_grad()
    def val_step(self, model_w, val_loader, device):
        model_w.eval()
        if val_loader is None:
            return None
        for batch in val_loader:
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            model_w.on_val_step(batch)
            if self.eval_data_back_to_cpu:
                move_to_device(batch, "cpu")
        return model_w.collect_notes()

    # @torch.no_grad()
    def test_step(self, model_w, test_loader, device):
        model_w.eval()
        if test_loader is None:
            return None
        for batch in test_loader:
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            model_w.on_test_step(batch)
            if self.eval_data_back_to_cpu:
                move_to_device(batch, "cpu")
        return model_w.collect_notes()
