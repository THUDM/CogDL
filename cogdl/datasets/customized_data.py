import os

import torch
from sklearn.preprocessing import StandardScaler

from cogdl.data import Dataset, Graph, MultiGraphDataset
from cogdl.utils import Accuracy, MultiLabelMicroF1, MultiClassMicroF1, CrossEntropyLoss, BCEWithLogitsLoss


def _get_evaluator(metric):
    if metric == "accuracy":
        return Accuracy()
    elif metric == "multilabel_f1":
        return MultiLabelMicroF1()
    elif metric == "multiclass_f1":
        return MultiClassMicroF1()
    else:
        raise NotImplementedError


def _get_loss_fn(metric):
    if metric in ("accuracy", "multiclass_f1"):
        return CrossEntropyLoss()
    elif metric == "multilabel_f1":
        return BCEWithLogitsLoss()
    else:
        raise NotImplementedError


def scale_feats(data):
    scaler = StandardScaler()
    feats = data.x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    data.x = feats
    return data


def generate_random_graph(num_nodes=100, num_edges=1000, num_feats=64):
    # load or generate your dataset
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    x = torch.randn(num_nodes, num_feats)
    y = torch.randint(0, 2, (num_nodes,))

    # set train/val/test mask in node_classification task
    train_mask = torch.zeros(num_nodes).bool()
    train_mask[0 : int(0.3 * num_nodes)] = True
    val_mask = torch.zeros(num_nodes).bool()
    val_mask[int(0.3 * num_nodes) : int(0.7 * num_nodes)] = True
    test_mask = torch.zeros(num_nodes).bool()
    test_mask[int(0.7 * num_nodes) :] = True
    data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data


class NodeDataset(Dataset):
    """
    data_path : path to load dataset. The dataset must be processed to specific format
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self, path="data.pt", data=None, scale_feat=True, metric="auto"):
        self.path = path
        self.data = data
        super(NodeDataset, self).__init__(root=path)
        if self.data is None:
            try:
                self.data = torch.load(path)
            except Exception as e:
                print(e)
                exit(1)
        if scale_feat:
            self.data = scale_feats(self.data)
        self.metric = metric
        if hasattr(self.data, "y") and self.data.y is not None:
            if metric == "auto":
                if len(self.data.y.shape) > 1:
                    self.metric = "multilabel_f1"
                else:
                    self.metric = "accuracy"

    def download(self):
        pass

    def process(self):
        if self.data is None:
            raise NotImplementedError
        return self.data

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
            torch.save(data, self.path)

    def __repr__(self):
        return "{}".format(self.path)


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
        if hasattr(self.data, "y") and self.data.y is not None:
            if metric == "auto":
                if len(self.data.y.shape) > 1:
                    self.metric = "multilabel_f1"
                else:
                    self.metric = "accuracy"

    def _download(self):
        pass

    def process(self):
        raise NotImplementedError

    def _process(self):
        if not os.path.exists(self.path):
            data = self.process()
            torch.save(data, self.path)

    def get_evaluator(self):
        return _get_evaluator(self.metric)

    def get_loss_fn(self):
        return _get_loss_fn(self.metric)

    def __repr__(self):
        return "{}".format(self.path)
