import copy
import os

import torch
import torch.nn as nn
import numpy as np

from .. import ModelWrapper
from cogdl.wrappers.tools.memory_moco import MemoryMoCo, NCESoftmaxLoss, moment_update
from cogdl.utils.optimizer import LinearOptimizer

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp


class GCCModelWrapper(ModelWrapper):
    @staticmethod
    def add_args(parser):
        # loss function
        parser.add_argument("--nce-k", type=int, default=16384)
        parser.add_argument("--nce-t", type=float, default=0.07)
        parser.add_argument("--finetune", action="store_true")
        parser.add_argument("--pretrain", action="store_true")
        parser.add_argument("--freeze", action="store_true")
        parser.add_argument("--momentum", type=float, default=0.999)

        # specify folder
        parser.add_argument("--save-model-path", type=str, default="saved", help="path to save model")
        parser.add_argument("--load-model-path", type=str, default="", help="path to load model")

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
        num_shuffle=10,
        save_model_path="saved",
        load_model_path="",
        freeze=False,
        pretrain=False
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

        self.num_shuffle = num_shuffle
        self.finetune = finetune
        self.pretrain = pretrain
        self.freeze = freeze
        self.save_model_path = save_model_path
        self.load_model_path = load_model_path
        
        if finetune:
            self.linear = nn.Linear(self.output_size, num_classes)
        else:
            self.register_buffer("linear", None)

    def train_step(self, batch):
        if self.finetune:
            return self.train_step_finetune(batch)
        elif self.pretrain:
            self.model_ema.eval()

            def set_bn_train(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.train()
            self.model_ema.apply(set_bn_train)
            return self.train_step_pretraining(batch)
        elif self.freeze:
            pass

    def train_step_pretraining(self, batch):
        # out = self.train_step_freeze(batch)
        graph_q, graph_k = batch
        
        # ===================Moco forward=====================
        feat_q = self.model(graph_q)
        with torch.no_grad():
            feat_k = self.model_ema(graph_k)

        out = self.contrast(feat_q, feat_k)
        assert feat_q.shape == (graph_q.batch_size, self.output_size)
        moment_update(self.model, self.model_ema, self.momentum)

        loss = self.criterion(out,)
        return loss

    def train_step_finetune(self, batch):
        graph, y = batch
        hidden = self.model(graph)
        pred = self.linear(hidden)
        # loss = self.default_loss_fn(pred, y)
        loss = self.criterion(pred, y)
        return loss
    
    def ge_step(self, batch):
        graph_q, graph_k = batch
        with torch.no_grad():
            feat_q = self.model(graph_q)
            feat_k = self.model(graph_k)
        bsz = graph_q.batch_size
        assert feat_q.shape == (bsz, self.output_size)
        emb = ((feat_q + feat_k) / 2).detach().cpu()
        return emb

    def test_step(self, batch):
        # assert self.load_emb_path
        if self.freeze:
            graph_q, graph_k, y = batch
            embeddings = self.ge_step((graph_q, graph_k))
            
            if len(y.shape) == 1: 
                num_classes = y.max().cpu().item() + 1
                y = nn.functional.one_hot(y, num_classes)
        
            dic_results = evaluate_nc(embeddings, y.cpu(), self.num_shuffle)
            self.note("Micro-F1_mean", dic_results["Micro-F1_mean"])
        elif self.finetune:
            self.linear.eval()
            graph_q, y = batch
            bsz = graph_q.batch_size
            with torch.no_grad():
                feat_q = self.model(graph_q)
                assert feat_q.shape == (bsz, self.output_size)
                out = self.linear(feat_q)
            # loss = self.criterion(out, y)
            preds = out.argmax(dim=1)
            f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")
            self.note("Micro-F1", f1)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        lr = cfg["lr"]
        weight_decay = cfg["weight_decay"]
        warm_steps = cfg["n_warmup_steps"]
        epochs = cfg["epochs"]
        batch_size = cfg["batch_size"]
        if "betas" in cfg:
            betas = cfg["betas"]
        else:
            betas = None
        total = cfg["total"]
        if warm_steps > 0 and warm_steps < 1:
            warm_steps = warm_steps * total

        if self.finetune:
            optimizer = torch.optim.Adam(
                [{"params": self.model.parameters()}, {"params": self.linear.parameters()}],
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas if betas else (0.9, 0.999))
        optimizer = LinearOptimizer(optimizer, warm_steps, epochs * (total // batch_size), init_lr=lr)
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
        if self.freeze or self.finetune:
            self.load_checkpoint(self.load_model_path)
            if self.finetune:
                self.model.apply(clear_bn)

    def post_stage(self, stage, data_w):
        if self.pretrain:
            filepath = os.path.join(self.save_model_path, "gcc_pretrain.pt")
            self.save_checkpoint(filepath)
        else:
            pass


def clear_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.reset_running_stats()


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels


def evaluate_nc(features_matrix, label_matrix, num_shuffle):
    # shuffle, to create train/test groups
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = []
    labels = label_matrix.argmax(axis=1).squeeze().tolist()
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)

    # score each train/test group
    all_results = defaultdict(list)

    for train_idx, test_idx in idx_list:

        X_train = features_matrix[train_idx]
        y_train = label_matrix[train_idx]

        X_test = features_matrix[test_idx]
        y_test = label_matrix[test_idx]

        clf = TopKRanker(LogisticRegression(solver='liblinear', C=1000))  # max_iter=1000
        clf.fit(X_train, y_train)

        # find out how many labels should be predicted
        top_k_list = y_test.sum(axis=1).long().tolist()
        preds = clf.predict(X_test, top_k_list)
        result = f1_score(y_test, preds, average="micro")
        all_results[""].append(result)
    # return "Micro-F1_mean", sum(all_results.values())/len(all_results)

    return dict(
        ("Micro-F1_mean", sum(all_results[train_percent]) / len(all_results[train_percent]),)
        for train_percent in sorted(all_results.keys())
    )