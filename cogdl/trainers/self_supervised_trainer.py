import os
import copy
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from .base_trainer import BaseTrainer
from . import register_trainer

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


class SelfSupervisedBaseTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--subgraph-sampling', action='store_true')
        parser.add_argument('--sample-size', type=int, default=8192)
        # fmt: on

    def __init__(self, args):
        super(SelfSupervisedBaseTrainer, self).__init__()
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.epochs = args.max_epoch
        self.patience = args.patience
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.dataset_name = args.dataset
        self.model_name = args.model
        self.sampling = args.subgraph_sampling
        self.sample_size = args.sample_size

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def fit(self, model, dataset):
        raise NotImplementedError


@register_trainer("self_supervised_joint")
class SelfSupervisedJointTrainer(SelfSupervisedBaseTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        SelfSupervisedBaseTrainer.add_args(parser)
        parser.add_argument('--alpha', default=10, type=float)
        # fmt: on

    def __init__(self, args):
        super(SelfSupervisedJointTrainer, self).__init__(args)
        self.alpha = args.alpha

    def fit(self, model, dataset):
        self.data = dataset.data
        self.data.add_remaining_self_loops()
        self.model = model
        if hasattr(self.model, "generate_virtual_labels"):
            self.model.generate_virtual_labels(self.data)
        self.set_loss_eval(dataset)
        self.data.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.to(self.device)
        epoch_iter = tqdm(range(self.epochs))

        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf

        for epoch in epoch_iter:
            aux_loss = self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Aux loss: {aux_loss:.4f}"
            )
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss.cpu()))
                max_score = np.max((max_score, val_acc))
        print(f"Valid accurracy = {best_score}")

        return best_model

    def set_loss_eval(self, dataset):
        self.loss_fn = dataset.get_loss_fn()
        self.evaluator = dataset.get_evaluator()

    def _train_step(self):
        data = self.model.transform_data() if hasattr(self.model, "transform_data") else self.data
        if self.sampling:
            data = data.to("cpu")
            idx = np.random.choice(np.arange(self.data.num_nodes), self.sample_size, replace=False)
            data = data.subgraph(idx).to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model.node_classification_loss(data)
        self_supervised_loss = self.model.self_supervised_loss(data)
        loss = loss + self.alpha * self_supervised_loss
        loss.backward()
        self.optimizer.step()

        return self_supervised_loss

    def _test_step(self, split="train"):
        self.model.eval()
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        with torch.no_grad():
            logits = self.model.predict(self.data)
        loss = self.loss_fn(logits[mask], self.data.y[mask])
        metric = self.evaluator(logits[mask], self.data.y[mask])
        return metric, loss


@register_trainer("self_supervised_pt_ft")
class SelfSupervisedPretrainer(SelfSupervisedBaseTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        SelfSupervisedBaseTrainer.add_args(parser)
        parser.add_argument('--alpha', default=1, type=float)
        parser.add_argument('--save-dir', default="./embedding", type=str)
        parser.add_argument('--load-dir', default="./embedding", type=str)
        parser.add_argument('--do-train', action='store_true')
        parser.add_argument('--do-eval', action='store_true')
        parser.add_argument('--eval-agc', action='store_true')
        # fmt: on

    def __init__(self, args):
        super(SelfSupervisedPretrainer, self).__init__(args)
        self.dataset_name = args.dataset
        self.model_name = args.model
        self.alpha = args.alpha
        self.save_dir = args.save_dir
        self.load_dir = args.load_dir
        self.do_train = args.do_train
        self.do_eval = args.do_eval
        self.eval_agc = args.eval_agc

    def fit(self, model, dataset):
        self.data = dataset.data
        self.data.add_remaining_self_loops()
        self.model = None

        if self.do_train:
            best = 1e9
            cnt_wait = 0
            self.model = copy.deepcopy(model)
            if hasattr(self.model, "generate_virtual_labels"):
                self.model.generate_virtual_labels(self.data)

            self.data = self.data.to(self.device)
            self.model = self.model.to(self.device)

            optimizer = torch.optim.Adam(
                self.model.get_parameters() if hasattr(self.model, "get_parameters") else self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            epoch_iter = tqdm(range(self.epochs))

            self.model.train()
            for epoch in epoch_iter:
                optimizer.zero_grad()
                data = self.model.transform_data() if hasattr(self.model, "transform_data") else self.data
                if self.sampling:
                    data = data.to("cpu")
                    idx = np.random.choice(np.arange(self.data.num_nodes), self.sample_size, replace=False)
                    data = data.subgraph(idx).to(self.device)

                loss = self.alpha * self.model.self_supervised_loss(data)
                epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss.item() / self.alpha: .4f}")

                if loss < best:
                    best = loss
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                if cnt_wait == self.patience:
                    print("Early stopping!")
                    break

                loss.backward()
                optimizer.step()

            self.model = self.model.to("cpu")
            if hasattr(self.model, "device"):
                self.model.device = "cpu"
            if self.save_dir is not None:
                with torch.no_grad():
                    embeds = self.model.embed(self.data.to("cpu"))
                self.save_embed(embeds)

        if self.do_eval:
            embeds = None
            if self.model is not None:
                with torch.no_grad():
                    embeds = self.model.embed(self.data.to("cpu"))
            else:
                embeds = np.load(os.path.join(self.load_dir, f"{self.model_name}_{self.dataset_name}.npy"))
                embeds = torch.from_numpy(embeds).to(self.device)

            if self.eval_agc:
                nclass = int(torch.max(self.data.y.cpu()) + 1)
                kmeans = KMeans(n_clusters=nclass, random_state=0).fit(embeds.detach().cpu().numpy())
                clusters = kmeans.labels_
                print("cluster NMI: %.4lf" % (normalized_mutual_info_score(clusters, self.data.y.cpu())))

            return self.evaluate(embeds.detach(), dataset.get_loss_fn(), dataset.get_evaluator())

    def evaluate(self, embeds, loss_fn=None, evaluator=None):
        nclass = int(torch.max(self.data.y) + 1)
        opt = {
            "idx_train": self.data.train_mask.to(self.device),
            "idx_val": self.data.val_mask.to(self.device),
            "idx_test": self.data.test_mask.to(self.device),
            "num_classes": nclass,
        }
        result = LogRegTrainer().train(embeds, self.data.y.to(self.device), opt, loss_fn, evaluator)
        print(f"TestAcc: {result: .4f}")
        return dict(Acc=result)

    def save_embed(self, embed):
        os.makedirs(self.save_dir, exist_ok=True)
        embed = embed.cpu().numpy()
        out_file = os.path.join(self.save_dir, f"{self.model_name}_{self.dataset_name}.npy")
        np.save(out_file, embed)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class LogRegTrainer(object):
    def train(self, data, labels, opt, loss_fn=None, evaluator=None):
        device = data.device
        idx_train = opt["idx_train"].to(device)
        idx_test = opt["idx_test"].to(device)
        nclass = opt["num_classes"]
        nhid = data.shape[-1]
        labels = labels.to(device)

        train_embs = data[idx_train]
        test_embs = data[idx_test]

        train_lbls = labels[idx_train]
        test_lbls = labels[idx_test]
        tot = 0

        xent = nn.CrossEntropyLoss() if loss_fn is None else loss_fn

        for _ in range(50):
            log = LogReg(nhid, nclass).to(device)
            optimizer = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.to(device)

            for _ in range(100):
                log.train()
                optimizer.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                optimizer.step()

            logits = log(test_embs)
            if evaluator is None:
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            else:
                acc = evaluator(logits, test_lbls)
            tot += acc
        return tot / 50
