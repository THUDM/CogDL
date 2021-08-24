import numpy as np
import scipy.sparse as sp
from collections import defaultdict

from sklearn.utils import shuffle as skshuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

from .. import register_model_wrapper, EmbeddingModelWrapper


@register_model_wrapper("network_embedding_mw")
class NetworkEmbeddingModelWrapper(EmbeddingModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-shuffle", type=int, default=10)
        parser.add_argument("--training-percents", default=[0.9], type=float, nargs="+")
        # parser.add_argument("--enhance", type=str, default=None, help="use prone or prone++ to enhance embedding")
        # fmt: on

    def __init__(self, model, num_shuffle=1, training_percents=[0.1]):
        super(NetworkEmbeddingModelWrapper, self).__init__()
        self.model = model
        self.num_shuffle = num_shuffle
        self.training_percents = training_percents

    def train_step(self, batch):
        emb = self.model(batch)
        return emb

    def test_step(self, batch):
        x, y = batch
        return evaluate_node_embeddings(x, y, self.num_shuffle, self.training_percents)


def evaluate_node_embeddings(features_matrix, label_matrix, num_shuffle, training_percents):
    if len(label_matrix.shape) > 1:
        labeled_nodes = np.nonzero(np.sum(label_matrix, axis=1) > 0)[0]
        features_matrix = features_matrix[labeled_nodes]
        label_matrix = label_matrix[labeled_nodes]

    # shuffle, to create train/test groups
    shuffles = []
    for _ in range(num_shuffle):
        shuffles.append(skshuffle(features_matrix, label_matrix))

    # score each train/test group
    all_results = defaultdict(list)

    for train_percent in training_percents:
        for shuf in shuffles:
            X, y = shuf

            training_size = int(train_percent * len(features_matrix))
            X_train = X[:training_size, :]
            y_train = y[:training_size, :]

            X_test = X[training_size:, :]
            y_test = y[training_size:, :]

            clf = TopKRanker(LogisticRegression(solver="liblinear"))
            clf.fit(X_train, y_train)

            # find out how many labels should be predicted
            top_k_list = y_test.sum(axis=1).astype(np.int).tolist()
            preds = clf.predict(X_test, top_k_list)
            result = f1_score(y_test, preds, average="micro")
            all_results[train_percent].append(result)

    return dict(
        (f"Micro-F1 {train_percent}", np.mean(all_results[train_percent]))
        for train_percent in sorted(all_results.keys())
    )


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels
