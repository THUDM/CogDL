import torch

from cogdl.data import Dataset, MultiGraphDataset, Batch
from cogdl.utils import download_url, accuracy, multilabel_f1, multiclass_f1, bce_with_logits_loss, cross_entropy_loss


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


class BaseDataset(Dataset):
    def __init__(self, root=None):
        super(BaseDataset, self).__init__("custom")

    def process(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        return self.data

    def __len__(self):
        if hasattr(self, "y"):
            return len(self.y)
        elif hasattr(self, "x"):
            return self.x.shape[0]
        else:
            raise NotImplementedError


class CustomizedNodeClassificationDataset(BaseDataset):
    """
    data_path : path to load dataset. The dataset must be processed to specific format
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self, data_path, metric="accuracy"):
        super(CustomizedNodeClassificationDataset, self).__init__(root=data_path)
        try:
            self.data = torch.load(data_path)
        except Exception as e:
            print(e)
            exit(1)
        self.metric = metric
        if hasattr(self.data, "y"):
            if len(self.data.y.shape) > 1:
                self.metric = "multilabel_f1"

    def download(self):
        for name in self.raw_file_names:
            download_url("{}{}&dl=1".format(self.url, name), self.raw_dir, name=name)

    def process(self):
        pass

    def get(self, idx):
        assert idx == 0
        return self.data

    def get_evaluator(self):
        return _get_evaluator(self.metric)

    def get_loss_fn(self):
        return _get_loss_fn(self.metric)

    def __repr__(self):
        return "{}()".format(self.name)


class CustomizedGraphClassificationDataset(BaseDataset):
    def __init__(self, data_path, metric="accuracy"):
        super(CustomizedGraphClassificationDataset, self).__init__(root=data_path)
        try:
            data = torch.load(data_path)
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

    def get_evaluator(self):
        return _get_evaluator(self.metric)

    def get_loss_fn(self):
        return _get_loss_fn(self.metric)

    def __repr__(self):
        return "{}()".format(self.name)
