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


class BaseDataset(Dataset):
    def __init__(self, root=None):
        super(BaseDataset, self).__init__("custom")
        self.root = root

    def process(self):
        raise NotImplementedError

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.root):
            self.process()

    def get(self, idx):
        return self.data

    def __len__(self):
        if hasattr(self, "y"):
            return len(self.y)
        elif hasattr(self, "x"):
            return self.x.shape[0]
        else:
            raise NotImplementedError


class NodeDataset(Dataset):
    """
    data_path : path to load dataset. The dataset must be processed to specific format
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self, path, scale_feat=True, metric="accuracy"):
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
        if hasattr(self.data, "y"):
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
    def __init__(self, path, metric="accuracy"):
        super(GraphDataset, self).__init__(root=path)
        self.path = path
        try:
            data = torch.load(path)
            if hasattr(data, "y"):
                self.y = torch.cat([idata.y for idata in data])
            if isinstance(data, list):
                batch = Batch.from_data_list(data)
                self.data = batch
                self.slices = batch.__slices__
                del self.data.batch
            else:
                assert len(data) == 0
                self.data = data[0]
                self.slices = data[1]
        except Exception as e:
            print(e)
            exit(1)

        self.metric = metric
        if hasattr(self.data, "y"):
            if len(self.data.y.shape) > 1:
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
