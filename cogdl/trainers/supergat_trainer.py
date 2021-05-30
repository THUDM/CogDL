from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import numpy as np

from .base_trainer import BaseTrainer
from . import register_trainer
from cogdl.models.supervised_model import SupervisedModel
from cogdl.data import Dataset

from tqdm import tqdm
import copy


def is_pretraining(current_epoch, pretraining_epoch):
    return current_epoch is not None and pretraining_epoch is not None and current_epoch < pretraining_epoch


def get_supervised_attention_loss(model, criterion=None):
    loss_list = []
    cache_list = [(m, m.cache) for m in model.modules()]

    criterion = nn.BCEWithLogitsLoss() if criterion is None else eval(criterion)
    for i, (module, cache) in enumerate(cache_list):
        # Attention (X)
        att = cache["att_with_negatives"]  # [E + neg_E, heads]
        # Labels (Y)
        label = cache["att_label"]  # [E + neg_E]

        att = att.mean(dim=-1)  # [E + neg_E]
        loss = criterion(att, label)
        loss_list.append(loss)

    return sum(loss_list)


def mix_supervised_attention_loss_with_pretraining(
    loss, model, mixing_weight, criterion=None, current_epoch=None, pretraining_epoch=None
):
    if mixing_weight == 0:
        return loss

    current_pretraining = is_pretraining(current_epoch, pretraining_epoch)
    next_pretraining = is_pretraining(current_epoch + 1, pretraining_epoch)

    for m in model.modules():
        current_pretraining = current_pretraining if m.pretraining is not None else None
        m.pretraining = next_pretraining if m.pretraining is not None else None

    if (current_pretraining is None) or (not current_pretraining):
        w1, w2 = 1.0, mixing_weight  # Forbid pre-training or normal-training
    else:
        w1, w2 = 0.0, 1.0  # Pre-training

    loss = w1 * loss + w2 * get_supervised_attention_loss(
        model=model,
        criterion=criterion,
    )
    return loss


class SuperGATTrainer(BaseTrainer):
    def __init__(self, args):
        super(SuperGATTrainer, self).__init__()
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.epochs = args.max_epoch
        self.patience = args.patience
        self.num_classes = args.num_classes
        self.hidden_size = args.hidden_size
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.val_interval = args.val_interval
        self.att_lambda = args.att_lambda
        self.total_pretraining_epoch = args.total_pretraining_epoch

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def fit(self, model: SupervisedModel, dataset: Dataset):
        self.data = dataset[0]
        self.model = model
        self.evaluator = dataset.get_evaluator()
        self.loss_fn = dataset.get_loss_fn()

        self.data.to(self.device)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        epoch_iter = tqdm(range(self.epochs))

        val_loss = 0
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        for epoch in epoch_iter:
            self._train_step(epoch)
            if epoch % self.val_interval == 0:
                train_acc, _ = self._test_step(split="train")
                val_acc, val_loss = self._test_step(split="val")
                test_acc, _ = self._test_step(split="test")
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
                )
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_loss <= best_loss:  # and val_acc >= best_score:
                        best_loss = val_loss
                        best_score = val_acc
                        test_score = test_acc
                    min_loss = np.min((min_loss, val_loss.cpu()))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
        return dict(Acc=test_score, ValAcc=best_score)

    def _train_step(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        # Forward
        outputs = self.model(
            self.data.x,
            self.data.edge_index,
            batch=None,
            attention_edge_index=getattr(self.data, "edge_index_train", None),
        )

        # Loss
        loss = self.loss_fn(outputs[self.data.train_mask], self.data.y[self.data.train_mask])
        # Supervision Loss w/ pretraining
        loss = mix_supervised_attention_loss_with_pretraining(
            loss=loss,
            model=self.model,
            mixing_weight=self.att_lambda,
            criterion=None,
            current_epoch=epoch,
            pretraining_epoch=self.total_pretraining_epoch,
        )
        loss.backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                self.data.x,
                self.data.edge_index,
                batch=None,
                attention_edge_index=getattr(self.data, "{}_edge_index".format(split), None),
            )
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        loss = self.loss_fn(logits[mask], self.data.y[mask])
        metric = self.evaluator(logits[mask], self.data.y[mask])
        return metric, loss
