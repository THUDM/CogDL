from typing import Union, Callable
import numpy as np
import warnings
from cogdl import function as BF
from cogdl.backend import BACKEND

if BACKEND == "jittor":
    from jittor import nn
    from jittor import nn as F
elif BACKEND == "torch":
    import torch.nn as nn
    import torch.nn.functional as F
else:
    raise ("Unsupported backend:", BACKEND)

from sklearn.metrics import f1_score


def setup_evaluator(metric: Union[str, Callable]):
    if isinstance(metric, str):
        metric = metric.lower()
        if metric == "acc" or metric == "accuracy":
            return Accuracy()
        elif metric == "multilabel_microf1" or "microf1" or "micro_f1":
            return MultiLabelMicroF1()
        elif metric == "multiclass_microf1":
            return MultiClassMicroF1()
        else:
            raise NotImplementedError
    else:
        return BaseEvaluator(metric)


class BaseEvaluator(object):
    def __init__(self, eval_func):
        self.y_pred = list()
        self.y_true = list()
        self.eval_func = eval_func

    def __call__(self, y_pred, y_true):
        metric = self.eval_func(y_pred, y_true)
        self.y_pred.append(BF.cpu(y_pred))
        self.y_true.append(BF.cpu(y_true))
        return metric

    def clear(self):
        self.y_pred = list()
        self.y_true = list()

    def evaluate(self):
        if len(self.y_pred) > 0:
            y_pred = BF.cat(self.y_pred, dim=0)
            y_true = BF.cat(self.y_true, dim=0)
            self.clear()
            return self.eval_func(y_pred, y_true)
        return 0


class MAE(object):
    def __init__(self):
        super(MAE, self).__init__()
        self.MAE = list()

    def __call__(self, y_pred, y_true):

        d = np.abs(y_true - y_pred)
        mae = d.tolist()
        MAE = np.array(mae).mean()
        self.MAE.append(MAE)
        return MAE

    def evaluate(self):
        if len(self.MAE) > 0:
            return np.sum(self.MAE) / len(self.MAE)
        warnings.warn("pre-computing list is empty")
        return 0

    def clear(self):
        self.MAE = list()


class Accuracy(object):
    def __init__(self, mini_batch=False):
        super(Accuracy, self).__init__()
        self.mini_batch = mini_batch
        self.tp = list()
        self.total = list()

    def __call__(self, y_pred, y_true):
        pred = (BF.argmax(y_pred, 1) == y_true).int()
        tp = pred.sum().int()
        total = pred.shape[0]
        if BF.is_tensor(tp):
            tp = tp.item()

        # if self.mini_batch:
        self.tp.append(tp)
        self.total.append(total)

        return tp / total

    def evaluate(self):
        if len(self.tp) > 0:
            tp = np.sum(self.tp)
            total = np.sum(self.total)
            self.tp = list()
            self.total = list()
            return tp / total
        warnings.warn("pre-computing list is empty")
        return 0

    def clear(self):
        self.tp = list()
        self.total = list()


class MultiLabelMicroF1(Accuracy):
    def __init__(self, mini_batch=False):
        super(MultiLabelMicroF1, self).__init__(mini_batch)

    def __call__(self, y_pred, y_true, sigmoid=False):
        if sigmoid:
            border = 0.5
        else:
            border = 0
        y_pred[y_pred >= border] = 1
        y_pred[y_pred < border] = 0
        tp = BF.to((y_pred * y_true).sum(), BF.dtype_dict("float32")).item()
        fp = BF.to(((1 - y_true) * y_pred).sum(), BF.dtype_dict("float32")).item()
        fn = BF.to((y_true * (1 - y_pred)).sum(), BF.dtype_dict("float32")).item()
        total = tp + fp + fn

        # if self.mini_batch:
        self.tp.append(int(tp))
        self.total.append(int(total))

        if total == 0:
            return 0
        return float(tp / total)


class MultiClassMicroF1(Accuracy):
    def __init__(self, mini_batch=False):
        super(MultiClassMicroF1, self).__init__(mini_batch)


class CrossEntropyLoss(nn.Module):
    def __call__(self, y_pred, y_true):
        y_true = y_true.long()
        y_pred = F.log_softmax(y_pred, dim=-1)
        return F.nll_loss(y_pred, y_true)


class BCEWithLogitsLoss(nn.Module):
    def __call__(self, y_pred, y_true, reduction="mean"):
        y_true = y_true.float()
        loss = nn.BCEWithLogitsLoss(reduction=reduction)(y_pred, y_true)
        if reduction == "none":
            loss = BF.sum(BF.mean(loss, dim=0))
        return loss


def multilabel_f1(y_pred, y_true, sigmoid=False):
    if sigmoid:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    tp = BF.to((y_true * y_pred).sum(), BF.dtype_dict("float32"))
    # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = BF.to(((1 - y_true) * y_pred).sum(), BF.dtype_dict("float32"))
    fn = BF.to((y_true * (1 - y_pred)).sum(), BF.dtype_dict("float32"))

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)
    return f1.item()


def multiclass_f1(y_pred, y_true):
    y_true = BF.squeeze(y_true).long()
    preds = BF.argmax(y_pred, 1)
    preds = BF.cpu(preds).detach().numpy()
    labels = BF.cpu(y_true).detach().numpy()
    micro = f1_score(labels, preds, average="micro")
    return micro


def accuracy(y_pred, y_true):
    y_true = BF.squeeze(y_true).long()
    preds = BF.type_as(BF.argmax(y_pred, 1), y_true)
    correct = BF.eq(preds, y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def cross_entropy_loss(y_pred, y_true):
    y_true = y_true.long()
    y_pred = F.log_softmax(y_pred, dim=-1)
    return F.nll_loss(y_pred, y_true)


def bce_with_logits_loss(y_pred, y_true, reduction="mean"):
    y_true = y_true.float()
    loss = nn.BCEWithLogitsLoss(reduction=reduction)(y_pred, y_true)
    if reduction == "none":
        loss = BF.sum(BF.mean(loss, dim=0))
    return loss
