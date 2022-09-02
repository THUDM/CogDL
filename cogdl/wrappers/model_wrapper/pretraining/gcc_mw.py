import copy

import torch
import torch.nn as nn

from .. import ModelWrapper
from cogdl.wrappers.tools.memory_moco import MemoryMoCo, NCESoftmaxLoss, moment_update
from cogdl.utils.optimizer import LinearOptimizer


class GCCModelWrapper(ModelWrapper):
    @staticmethod
    def add_args(parser):
        # loss function
        parser.add_argument("--nce-k", type=int, default=32)
        parser.add_argument("--nce-t", type=float, default=0.07)
        parser.add_argument("--finetune", action="store_true")
        parser.add_argument("--momentum", type=float, default=0.96)

        # specify folder
        parser.add_argument("--model-path", type=str, default="gcc_pretrain.pt", help="path to save model")

    def __init__(
        self,
        model,
        optimizer_cfg,
        nce_k,
        nce_t,
        momentum,
        output_size,
        finetune=False,
        num_classes=1,
        model_path="gcc_pretrain.pt",
    ):
        super(GCCModelWrapper, self).__init__()
        self.model = model
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.detach_()

        self.optimizer_cfg = optimizer_cfg
        self.output_size = output_size
        self.momentum = momentum

        self.contrast = MemoryMoCo(self.output_size, num_classes, nce_k, nce_t, use_softmax=True)
        self.criterion = nn.CrossEntropyLoss() if finetune else NCESoftmaxLoss()

        self.finetune = finetune
        self.model_path = model_path
        if finetune:
            self.linear = nn.Linear(self.output_size, num_classes)
        else:
            self.register_buffer("linear", None)

    def train_step(self, batch):
        if self.finetune:
            return self.train_step_finetune(batch)
        else:
            return self.train_step_pretraining(batch)

    def train_step_pretraining(self, batch):
        graph_q, graph_k = batch

        # ===================Moco forward=====================
        feat_q = self.model(graph_q)
        with torch.no_grad():
            feat_k = self.model_ema(graph_k)

        out = self.contrast(feat_q, feat_k)

        assert feat_q.shape == (graph_q.batch_size, self.output_size)
        moment_update(self.model, self.model_ema, self.momentum)

        loss = self.criterion(
            out,
        )
        return loss

    def train_step_finetune(self, batch):
        graph = batch
        y = graph.y
        hidden = self.model(graph)
        pred = self.linear(hidden)
        loss = self.default_loss_fn(pred, y)
        return loss

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        lr = cfg["lr"]
        weight_decay = cfg["weight_decay"]
        warm_steps = cfg["n_warmup_steps"]
        epochs = cfg["epochs"]
        batch_size = cfg["batch_size"]
        if self.finetune:
            optimizer = torch.optim.Adam(
                [{"params": self.model.parameters()}, {"params": self.linear.parameters()}],
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = LinearOptimizer(optimizer, warm_steps, epochs * batch_size, init_lr=lr)
        return optimizer

    def save_checkpoint(self, path):
        state = {
            "model": self.model.state_dict(),
            "contrast": self.contrast.state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.model_ema.load_state_dict(state["model_ema"])
        self.contrast.load_state_dict(state["contrast"])

    def pre_stage(self, stage, data_w):
        if self.finetune:
            self.load_checkpoint(self.model_path)
            self.model.apply(clear_bn)

    def post_stage(self, stage, data_w):
        if not self.finetune:
            self.save_checkpoint(self.model_path)


def clear_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.reset_running_stats()
