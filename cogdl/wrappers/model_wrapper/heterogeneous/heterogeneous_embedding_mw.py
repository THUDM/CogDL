import argparse
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .. import EmbeddingModelWrapper


class HeterogeneousEmbeddingModelWrapper(EmbeddingModelWrapper):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    def __init__(self, model, hidden_size=200):
        super(HeterogeneousEmbeddingModelWrapper, self).__init__()
        self.model = model
        self.hidden_size = hidden_size

    def train_step(self, batch):
        embeddings = self.model(batch)
        embeddings = np.hstack((embeddings, batch.x.numpy()))

        return embeddings

    def test_step(self, batch):
        embeddings, data = batch

        # Select nodes which have label as training data
        train_index = torch.cat((data.train_node, data.valid_node)).numpy()
        test_index = data.test_node.numpy()
        y = data.y.numpy()

        X_train, y_train = embeddings[train_index], y[train_index]
        X_test, y_test = embeddings[test_index], y[test_index]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        test_f1 = f1_score(y_test, preds, average="micro")

        return dict(f1=test_f1)
