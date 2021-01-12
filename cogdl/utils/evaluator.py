import torch
from sklearn.metrics import f1_score


def multilabel_f1(y_true, y_pred, sigmoid=False):
    if sigmoid:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    preds = y_pred.cpu().detach()
    labels = y_true.cpu().float()
    return f1_score(labels, preds, average="micro")


def multiclass_f1(y_true, y_pred):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = y_true.cpu().detach().numpy()
    micro = f1_score(labels, preds, average="micro")
    return micro


def accuracy(y_true, y_pred):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum()
    return correct / len(y_true)


def multilabel_evaluator(reduction="none"):
    loss_func = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    metric = multilabel_f1
    return loss_func, metric


def accuracy_evaluator():
    loss_func = torch.nn.functional.nll_loss
    metric = accuracy
    return loss_func, metric


def multiclass_evaluator():
    loss_func = torch.nn.functional.nll_loss
    metric = multiclass_f1
    return loss_func, metric
