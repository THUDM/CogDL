import os
from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.data import Data, DataLoader, InMemoryDataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from . import BaseTask, register_task

@register_task("unsupervised_graph_classification")
class UnsupervisedGraphClassification(BaseTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-shuffle", type=int, default=10)


    def __init__(self, args):
        super(UnsupervisedGraphClassification, self).__init__(args)
        dataset = build_dataset(args)
        self.label = np.array([data.y for data in dataset])
        print(self.label)
        self.data = [
            Data(x=data.x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr, pos=data.pos).apply(lambda x:x.cuda())
            for data in dataset
        ]
        self.num_graphs = len(self.data)
        self.num_classes = dataset.num_classes
        self.label_matrix = np.zeros((self.num_graphs, self.num_classes))
        self.label_matrix[range(self.num_graphs), np.array([data.y for data in self.data], dtype=int)] = 1


        self.model = build_model(args)
        self.model_name = args.model
        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle
        self.save_dir = args.save_dir
        self.epochs = args.max_epoch

        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        # )

    def train(self):
        prediction = None
        for epoch in range(self.epochs):
            prediction, loss = self.model(self.data)
            if loss is None:
                break
        if prediction is not None:
            self.save_emb(prediction)
            return self._evaluate(prediction)

    def save_emb(self, embs):
        name = os.path.join(self.save_dir, self.model_name + '_emb.npy')
        np.save(name, embs)

    def _evaluate(self, embeddings):
        shuffles = []
        for _ in range(self.num_shuffle):
            shuffles.append(skshuffle(embeddings, self.label))
        all_results = defaultdict(list)
        training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]
        for training_percent in training_percents:
            for shuf in shuffles:
                training_size = int(training_percent * self.num_graphs)
                X, y = shuf
                X_train = X[:training_size, :]
                y_train = y[:training_size]

                X_test = X[training_size:, :]
                y_test = y[training_size:]

                clf = SVC()
                clf.fit(X_train, y_train)

                preds = clf.predict(X_test)
                accuracy = accuracy_score(y_test, preds)
                all_results[training_percent].append(accuracy)

        return dict(
            (
                f"Accuracy {train_percent}",
                sum(all_results[train_percent]) / len(all_results[train_percent]),
            )
            for train_percent in sorted(all_results.keys())
        )