from typing import Dict

import random
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle as skshuffle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

from cogdl.utils import accuracy, multilabel_f1


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


def merge_batch_indexes(values: list, method="mean"):
    # if key.endswith("loss"):
    #     result = sum(values)
    #     if torch.is_tensor(result):
    #         result = result.item()
    #     result = result / len(values)
    # elif key.endswith("eval_index"):
    #     if len(values) > 1:
    #         val = torch.stack(values)
    #         val = val.sum(0)
    #     else:
    #         val = values[0]
    #     fp = val[0]
    #     all_ = val.sum()
    #
    #     prefix = key[: key.find("eval_index")]
    #     if val.shape[0] == 2:
    #         _key = prefix + "acc"
    #     else:
    #         _key = prefix + "f1"
    #     result = (fp / all_).item()
    #     out_key = _key

    if isinstance(values[0], dict) or isinstance(values[0], tuple):
        return values
    elif method == "mean":
        return sum(values) / len(values)
    elif method == "sum":
        return sum(values)
    else:
        return sum(values)


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
        deg = graph.degrees().long()
        degrees.append(deg)
        max_degree = max(deg.max().item(), max_degree)

    max_degree = int(max_degree) + 1
    for i in range(len(data)):
        one_hot = F.one_hot(degrees[i], max_degree).float()
        data[i].x = one_hot.to(device)
    return data


def split_dataset(ndata, train_ratio, test_ratio):

    train_size = int(ndata * train_ratio)
    test_size = int(ndata * test_ratio)
    index = np.arange(ndata)
    random.shuffle(index)

    train_index = index[:train_size]
    test_index = index[-test_size:]
    if train_ratio + test_ratio == 1:
        val_index = None
    else:
        val_index = index[train_size:-test_size]
    return train_index, val_index, test_index


def evaluate_node_embeddings_using_logreg(data, labels, train_idx, test_idx, run=20):
    result = LogRegTrainer().train(data, labels, train_idx, test_idx, run=run)
    return result


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class LogRegTrainer(object):
    def train(self, data, labels, idx_train, idx_test, loss_fn=None, evaluator=None, run=20):
        device = data.device
        nhid = data.shape[-1]
        labels = labels.to(device)

        train_embs = data[idx_train]
        test_embs = data[idx_test]

        train_lbls = labels[idx_train]
        test_lbls = labels[idx_test]
        tot = 0

        num_classes = int(labels.max()) + 1

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss() if len(labels.shape) == 1 else nn.BCEWithLogitsLoss()

        if evaluator is None:
            evaluator = accuracy if len(labels.shape) == 1 else multilabel_f1

        for _ in range(run):
            log = LogReg(nhid, num_classes).to(device)
            optimizer = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.to(device)

            for _ in range(100):
                log.train()
                optimizer.zero_grad()

                logits = log(train_embs)
                loss = loss_fn(logits, train_lbls)

                loss.backward()
                optimizer.step()

            log.eval()
            with torch.no_grad():
                logits = log(test_embs)
            metric = evaluator(logits, test_lbls)

            tot += metric
        return tot / run


def evaluate_node_embeddings_using_liblinear(features_matrix, label_matrix, num_shuffle, training_percents):
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


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X=embeddings, y=labels)
    for train_index, test_index in kf.split(embeddings):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-2, 1e-1, 1]}
        svc = SVC()
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return dict(acc=test_f1, std=test_std)


def evaluate_clustering(features_matrix, labels, cluster_method, num_clusters, num_nodes, full=True):
    print("Clustering...")
    if cluster_method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features_matrix)
        clusters = kmeans.labels_
    else:
        clustering = SpectralClustering(n_clusters=num_clusters, assign_labels="discretize", random_state=0).fit(
            features_matrix
        )
        clusters = clustering.labels_

    print("Evaluating...")
    truth = labels.cpu().numpy()
    if full:
        mat = np.zeros([num_clusters, num_clusters])
        for i in range(num_nodes):
            mat[clusters[i]][truth[i]] -= 1
        _, row_idx = linear_sum_assignment(mat)
        acc = -mat[_, row_idx].sum() / num_nodes
        for i in range(num_nodes):
            clusters[i] = row_idx[clusters[i]]
        macro_f1 = f1_score(truth, clusters, average="macro")
        return dict(acc=acc, nmi=normalized_mutual_info_score(clusters, truth), macro_f1=macro_f1)
    else:
        return dict(nmi=normalized_mutual_info_score(clusters, truth))
