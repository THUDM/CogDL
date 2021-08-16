import random

import torch
import torch.nn.functional as F


def pre_evaluation_index(y_pred, y_true, sigmoid=False):
    """
    Pre-calculating diffusion matrix for mini-batch evaluation
    Return:
        torch.Tensor((tp, all)) for multi-class classification
        torch.Tensor((tp, fp, fn)) for multi-label classification
    """
    if len(y_true.shape) == 1:
        pred = (y_pred.argmax(1) == y_true).int()
        tp = pred.sum()
        fnp = pred.shape[0] - tp
        return torch.tensor((tp, fnp)).float()
    else:
        if sigmoid:
            border = 0.5
        else:
            border = 0
        y_pred[y_pred >= border] = 1
        y_pred[y_pred < border] = 0
        tp = (y_pred * y_true).sum().to(torch.float32)
        # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        return torch.tensor((tp, fp, fn))


def node_degree_as_feature(data):
    r"""
    Set each node feature as one-hot encoding of degree
    :param data: a list of class Data
    :return: a list of class Data
    """
    max_degree = 0
    degrees = []
    device = data[0].edge_index[0].device

    for graph in data:
        deg = graph.degrees()
        degrees.append(deg)
        max_degree = max(deg.max().item(), max_degree)

    max_degree = int(max_degree) + 1
    for i in range(len(data)):
        one_hot = F.one_hot(degrees[i], max_degree)
        data[i].x = one_hot.to(device)
    return data


def split_dataset(dataset, train_ratio, test_ratio):
    droplast = False

    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio)
    index = list(range(len(dataset)))
    random.shuffle(index)

    train_index = index[:train_size]
    test_index = index[-test_size:]

    bs = 1
    train_dataset = dict(dataset=[dataset[i] for i in train_index], batch_size=bs, drop_last=droplast)
    test_dataset = dict(dataset=[dataset[i] for i in test_index], batch_size=bs, drop_last=droplast)
    if train_ratio + test_ratio < 1:
        val_index = index[train_size:-test_size]
        valid_dataset = dict(dataset=[dataset[i] for i in val_index], batch_size=bs, drop_last=droplast)
    else:
        valid_dataset = None
    return train_dataset, valid_dataset, test_dataset
