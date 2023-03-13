import copy
import warnings
from typing import Optional
import numpy as np
from tqdm import tqdm
import os

import jittor
from cogdl.wrappers.data_wrapper.base_data_wrapper import DataWrapper
from cogdl.wrappers.model_wrapper.base_model_wrapper import ModelWrapper, EmbeddingModelWrapper
from .trainer_utils import (
    evaluation_comp,
    load_model,
    save_model,
    Printer,
)

# from cogdl.trainer.embed_trainer import EmbeddingTrainer
from .controller import DataController
from cogdl.loggers import build_logger
from cogdl.data import Graph

# from cogdl.utils.grb_utils import adj_preprocess, updateGraph, adj_to_tensor


class Trainer(object):
    def __init__(
        self,
        epochs: int,
        max_epoch: int = None,
        nstage: int = 1,
        cpu: bool = False,
        checkpoint_path: str = "./checkpoints/model.pt",
        resume_training: str = False,
        device_ids: Optional[list] = None,
        distributed_training: bool = False,
        distributed_inference: bool = False,
        master_addr: str = "localhost",
        master_port: int = 10086,
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
        return_model: bool = False,
        actnn: bool = False,
        fp16: bool = False,
        rp_ratio: int = 1,
        attack=None,
        attack_mode="injection",
        do_test: bool = True,
        do_valid: bool = True,
    ):
        self.epochs = epochs
        self.nstage = nstage
        self.patience = patience
        self.early_stopping = early_stopping
        self.eval_step = eval_step
        self.monitor = None
        self.evaluation_metric = None
        self.progress_bar = progress_bar

        if max_epoch is not None:
            warnings.warn("The max_epoch is deprecated and will be removed in the future, please use epochs instead!")
            self.epochs = max_epoch

        self.cpu = cpu
        self.world_size = 0
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training

        self.distributed_training = distributed_training
        self.distributed_inference = distributed_inference

        self.master_addr = master_addr
        self.master_port = master_port

        self.cpu_inference = cpu_inference

        self.return_model = return_model

        self.on_train_batch_transform = None
        self.on_eval_batch_transform = None
        self.clip_grad_norm = clip_grad_norm

        self.save_emb_path = save_emb_path
        self.load_emb_path = load_emb_path

        self.data_controller = DataController(distributed=self.distributed_training)

        self.logger = build_logger(logger, log_path, project)

        self.after_epoch_hooks = []
        self.pre_epoch_hooks = []
        self.training_end_hooks = []

        # if distributed_training:
        #     self.register_training_end_hook(ddp_end)
        #     self.register_out_epoch_hook(ddp_after_epoch)

        self.eval_data_back_to_cpu = False

        self.fp16 = fp16
        self.attack = attack
        self.attack_mode = attack_mode
        self.do_test = do_test
        self.do_valid = do_valid

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

    def run(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        # for network/graph embedding models
        # if isinstance(model_w, EmbeddingModelWrapper):
        #     return EmbeddingTrainer(self.save_emb_path, self.load_emb_path).run(model_w, dataset_w)

        print("Model Parameters:", sum(p.numel() for p in model_w.parameters()))

        # for deep learning models
        # set default loss_fn and evaluator for model_wrapper
        # mainly for in-cogdl setting
        model_w.default_loss_fn = dataset_w.get_default_loss_fn()
        model_w.default_evaluator = dataset_w.get_default_evaluator()
        model_w.set_evaluation_metric()

        if self.resume_training:
            model_w = load_model(model_w, self.checkpoint_path).to(self.devices[0])

        self.train(model_w, dataset_w)
        best_model_w = load_model(model_w, self.checkpoint_path)

        if self.return_model:
            return best_model_w.model

        final_test = self.evaluate(best_model_w, dataset_w)

        # clear the GPU memory
        dataset = dataset_w.get_dataset()

        return final_test

    def evaluate(self, model_w: ModelWrapper, dataset_w: DataWrapper, cpu=False):

        # disable `distributed` to inference once only
        self.distributed_training = False
        dataset_w.prepare_test_data()
        if self.do_valid:
            final_val = self.validate(model_w, dataset_w)
        else:
            final_val = {}
        if self.do_test:
            final_test = self.test(model_w, dataset_w)
        else:
            final_test = {}

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

    def build_optimizer(self, model_w):
        opt_wrap = model_w.setup_optimizer()
        if isinstance(opt_wrap, list) or isinstance(opt_wrap, tuple):
            assert len(opt_wrap) == 2
            optimizers, lr_schedulers = opt_wrap
        else:
            optimizers = opt_wrap
            lr_schedulers = None

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if lr_schedulers and not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]
        return optimizers, lr_schedulers

    def train(self, model_w, dataset_w):  # noqa: C901
        self.data_controller.prepare_data_wrapper(dataset_w)
        self.eval_data_back_to_cpu = dataset_w.data_back_to_cpu

        optimizers, lr_schedulers = self.build_optimizer(model_w)
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

        best_model_w = None

        patience = 0
        best_epoch = 0
        for stage in range(self.nstage):
            with jittor.no_grad():
                pre_stage_out = model_w.pre_stage(stage, dataset_w)
                dataset_w.pre_stage(stage, pre_stage_out)
                self.data_controller.training_proc_per_stage(dataset_w)

            if self.progress_bar == "epoch":
                epoch_iter = tqdm(range(1, self.epochs + 1))
                epoch_printer = Printer(epoch_iter.set_description, world_size=self.world_size)
            else:
                epoch_iter = range(1, self.epochs + 1)
                epoch_printer = Printer(print, world_size=self.world_size)

            self.logger.start()
            print_str_dict = dict()
            if self.attack is not None:
                graph = dataset_w.dataset.data
                graph_backup = copy.deepcopy(graph)
                graph0 = copy.deepcopy(graph)
                num_train = jittor.sum(graph.train_mask).item()
            for epoch in epoch_iter:
                for hook in self.pre_epoch_hooks:
                    hook(self)

                # inductive setting ..
                dataset_w.train()
                train_loader = dataset_w.on_train_wrapper()
                train_dataset = train_loader.get_dataset_from_loader()
                # if hasattr(train_dataset, "shuffle"):
                #     train_dataset.shuffle()
                training_loss = self.train_step(model_w, train_loader, optimizers, lr_schedulers)

                # if self.attack is not None:
                #     if self.attack_mode == "injection":
                #         graph0.test_mask = graph0.train_mask
                #     else:
                #         graph0.test_mask[torch.where(graph0.train_mask)[0].multinomial(int(num_train * 0.01))] = True
                #     graph_attack = self.attack.attack(model=model_w.model, graph=graph0, adj_norm_func=None)  # todo
                #     adj_attack = graph_attack.to_scipy_csr()
                #     features_attack = graph_attack.x
                #     adj_train = adj_preprocess(adj=adj_attack, adj_norm_func=None, device=rank)  # todo
                #     n_inject = graph_attack.num_nodes - graph.num_nodes
                #     updateGraph(graph, adj_train, features_attack)
                #     graph.edge_weight = torch.ones(graph.num_edges, device=rank)
                #     graph.train_mask = torch.cat((graph.train_mask, torch.zeros(n_inject, dtype=bool, device=rank)), 0)
                #     graph.val_mask = torch.cat((graph.val_mask, torch.zeros(n_inject, dtype=bool, device=rank)), 0)
                #     graph.test_mask = torch.cat((graph.test_mask, torch.zeros(n_inject, dtype=bool, device=rank)), 0)
                #     graph.y = torch.cat((graph.y, torch.zeros(n_inject, device=rank)), 0)
                #     graph.grb_adj = adj_to_tensor(adj_train).to(rank)
                print_str_dict["Epoch"] = epoch
                print_str_dict["train_loss"] = training_loss

                val_loader = dataset_w.on_val_wrapper()
                if self.do_valid is True:
                    if val_loader is not None and epoch % self.eval_step == 0:
                        # inductive setting ..
                        dataset_w.eval()
                        # do validation in inference device
                        val_result = self.validate(model_w, dataset_w)
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

                epoch_printer(print_str_dict)
                self.logger.note(print_str_dict, epoch)

                for hook in self.after_epoch_hooks:
                    hook(self)

            with jittor.no_grad():
                model_w.eval()
                post_stage_out = model_w.post_stage(stage, dataset_w)
                dataset_w.post_stage(stage, post_stage_out)

            if best_model_w is None:
                best_model_w = copy.deepcopy(model_w)
            if self.attack is not None:
                dataset_w.dataset.data = graph_backup

        save_model(best_model_w, self.checkpoint_path, best_epoch)

        for hook in self.training_end_hooks:
            hook(self)

    def validate(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        model_w.eval()
        dataset_w.eval()

        val_loader = dataset_w.on_val_wrapper()
        with jittor.no_grad():
            result = self.val_step(model_w, val_loader)

        return result

    def test(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        model_w.eval()
        dataset_w.eval()

        test_loader = dataset_w.on_test_wrapper()
        if model_w.training_type == "unsupervised":
            result = self.test_step(model_w, test_loader)
        else:
            with jittor.no_grad():
                result = self.test_step(model_w, test_loader)

        return result

    def train_step(self, model_w, train_loader, optimizers, lr_schedulers):
        model_w.train()
        losses = []

        if self.progress_bar == "iteration":
            train_loader = tqdm(train_loader)

        for batch in train_loader:
            if hasattr(batch, "train_mask") and batch.train_mask.sum().item() == 0:
                continue

            loss = model_w.on_train_step(batch)

            for optimizer in optimizers:
                optimizer.step(loss)
                optimizer.clip_grad_norm(self.clip_grad_norm)

            losses.append(loss.item())
        if lr_schedulers is not None:
            for lr_schedular in lr_schedulers:
                lr_schedular.step()

        return np.mean(losses)

    def val_step(self, model_w, val_loader):
        model_w.eval()
        if val_loader is None:
            return None
        for batch in val_loader:
            model_w.on_val_step(batch)
        return model_w.collect_notes()

    def test_step(self, model_w, test_loader):
        model_w.eval()
        if test_loader is None:
            return None
        for batch in test_loader:
            model_w.on_test_step(batch)
        return model_w.collect_notes()
