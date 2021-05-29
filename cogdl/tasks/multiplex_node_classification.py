import argparse
import warnings

import networkx as nx
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task

warnings.filterwarnings("ignore")


@register_task("multiplex_node_classification")
class MultiplexNodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(MultiplexNodeClassification, self).__init__(args)
        dataset = build_dataset(args) if dataset is None else dataset
        self.data = dataset[0]
        self.label_matrix = self.data.y
        self.num_nodes, self.num_classes = dataset.num_nodes, dataset.num_classes
        self.hidden_size = args.hidden_size
        self.model = build_model(args) if model is None else model
        self.args = args

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.model = self.model.to(self.device)

    def train(self):
        G = nx.DiGraph()
        row, col = self.data.edge_index
        G.add_edges_from(list(zip(row.numpy(), col.numpy())))
        # G.add_edges_from(self.data.edge_index.t().tolist())
        if self.args.model != "gcc":
            embeddings = self.model.train(G, self.data.pos.tolist())
        else:
            embeddings = self.model.train(self.data)
        embeddings = np.hstack((embeddings, self.data.x.numpy()))

        # Select nodes which have label as training data
        train_index = torch.cat((self.data.train_node, self.data.valid_node)).numpy()
        test_index = self.data.test_node.numpy()
        y = self.data.y.numpy()

        X_train, y_train = embeddings[train_index], y[train_index]
        X_test, y_test = embeddings[test_index], y[test_index]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        test_f1 = f1_score(y_test, preds, average="micro")

        return dict(f1=test_f1)
