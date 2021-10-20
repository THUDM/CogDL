import numpy as np

from .. import EmbeddingModelWrapper
from sklearn.metrics import roc_auc_score, f1_score, auc, precision_recall_curve


class EmbeddingLinkPredictionModelWrapper(EmbeddingModelWrapper):
    def __init__(self, model):
        super(EmbeddingLinkPredictionModelWrapper, self).__init__()
        self.model = model

    def train_step(self, graph):
        embeddings = self.model(graph)
        return embeddings

    def test_step(self, batch):
        embeddings, test_data = batch
        roc_auc, f1_score, pr_auc = evaluate(embeddings, test_data[0], test_data[1])
        print(f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}")
        return dict(ROC_AUC=roc_auc, PR_AUC=pr_auc, F1=f1_score)


def evaluate(embs, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        true_list.append(1)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    for edge in false_edges:
        true_list.append(0)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def get_score(embs, node1, node2, eps=1e-5):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + eps)
