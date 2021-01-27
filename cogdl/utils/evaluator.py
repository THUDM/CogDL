import torch
from sklearn.metrics import f1_score


def multilabel_f1(y_pred, y_true, sigmoid=False):
    if sigmoid:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    preds = y_pred.cpu().detach()
    labels = y_true.cpu().float()
    return f1_score(labels, preds, average="micro")


def multiclass_f1(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = y_true.cpu().detach().numpy()
    micro = f1_score(labels, preds, average="micro")
    return micro


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def cross_entropy_loss(y_pred, y_true):
    y_true = y_true.long()
    y_pred = torch.nn.functional.log_softmax(y_pred, dim=-1)
    return torch.nn.functional.nll_loss(y_pred, y_true)


def bce_with_logits_loss(y_pred, y_true, reduction="mean"):
    y_true = y_true.float()
    loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)(y_pred, y_true)
    if reduction == "none":
        loss = torch.sum(torch.mean(loss, dim=0))
    return loss
