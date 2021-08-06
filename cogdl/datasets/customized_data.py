import os

import torch
from sklearn.preprocessing import StandardScaler

from cogdl.data import Dataset, Batch, MultiGraphDataset
from cogdl.utils import accuracy, multilabel_f1, multiclass_f1, bce_with_logits_loss, cross_entropy_loss


def _get_evaluator(metric):
    if metric == "accuracy":
        return accuracy
    elif metric == "multilabel_f1":
        return multilabel_f1
    elif metric == "multiclass_f1":
        return multiclass_f1
    else:
        raise NotImplementedError


def _get_loss_fn(metric):
    if metric in ("accuracy", "multiclass_f1"):
        return cross_entropy_loss
    elif metric == "multilabel_f1":
        return bce_with_logits_loss
    else:
        raise NotImplementedError


def scale_feats(data):
    scaler = StandardScaler()
    feats = data.x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    data.x = feats
    return data


class NodeDataset(Dataset):
    """
    data_path : path to load dataset. The dataset must be processed to specific format
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self, path="cus_data.pt", scale_feat=True, metric="accuracy"):
        self.path = path
        super(NodeDataset, self).__init__(root=path)
        try:
            self.data = torch.load(path)
            if scale_feat:
                self.data = scale_feats(self.data)
        except Exception as e:
            print(e)
            exit(1)
        self.metric = metric
        if hasattr(self.data, "y") and self.data.y is not None:
            if len(self.data.y.shape) > 1:
                self.metric = "multilabel_f1"
            else:
                self.metric = "accuracy"

    def download(self):
        pass

    def process(self):
        raise NotImplementedError

    def get(self, idx):
        assert idx == 0
        return self.data

    def get_evaluator(self):
        return _get_evaluator(self.metric)

    def get_loss_fn(self):
        return _get_loss_fn(self.metric)

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.path):
            data = self.process()
            if not os.path.exists(self.path):
                torch.save(data, self.path)

    def __repr__(self):
        return "{}()".format(self.name)


class GraphDataset(MultiGraphDataset):
    def __init__(self, path="cus_graph_data.pt", metric="accuracy"):
        self.path = path
        super(GraphDataset, self).__init__(root=path)
        # try:
        data = torch.load(path)
        if hasattr(data[0], "y") and data[0].y is None:
            self.y = torch.cat([idata.y for idata in data])
        self.data = data

        self.metric = metric
        if hasattr(self, "y") and self.y is not None:
            if len(self.y.shape) > 1:
                self.metric = "multilabel_f1"

    def _download(self):
        pass

    def process(self):
        raise NotImplementedError

    def _process(self):
        if not os.path.exists(self.path):
            data = self.process()
            if not os.path.exists(self.path):
                torch.save(data, self.path)

    def get_evaluator(self):
        return _get_evaluator(self.metric)

    def get_loss_fn(self):
        return _get_loss_fn(self.metric)

    def __repr__(self):
        return "{}()".format(self.name)
