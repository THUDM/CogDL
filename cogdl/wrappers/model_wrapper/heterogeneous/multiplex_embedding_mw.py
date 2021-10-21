import argparse
import numpy as np
import torch
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

from cogdl.data import Graph
from .. import EmbeddingModelWrapper


def get_score(embs, node1, node2):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


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


def evaluate_multiplex(all_embs, test_data):
    total_roc_auc, total_f1_score, total_pr_auc = [], [], []
    for key in test_data.keys():
        embs = all_embs[key]
        roc_auc, f1_score, pr_auc = evaluate(embs, test_data[key][0], test_data[key][1])
        total_roc_auc.append(roc_auc)
        total_f1_score.append(f1_score)
        total_pr_auc.append(pr_auc)
    assert len(total_roc_auc) > 0
    roc_auc, f1_score, pr_auc = (
        np.mean(total_roc_auc),
        np.mean(total_f1_score),
        np.mean(total_pr_auc),
    )
    print(f"Test ROC-AUC = {roc_auc:.4f}, F1 = {f1_score:.4f}, PR-AUC = {pr_auc:.4f}")
    return dict(ROC_AUC=roc_auc, PR_AUC=pr_auc, F1=f1_score)


class MultiplexEmbeddingModelWrapper(EmbeddingModelWrapper):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=200)
        parser.add_argument("--eval-type", type=str, default='all', nargs='+')
        # fmt: on

    def __init__(self, model, hidden_size=200, eval_type="all"):
        super(MultiplexEmbeddingModelWrapper, self).__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.eval_type = eval_type

    def train_step(self, batch):
        if hasattr(self.model, "multiplicity"):
            all_embs = self.model(batch)
        else:
            all_embs = dict()
            for key in batch.keys():
                if self.eval_type == "all" or key in self.eval_type:
                    G = Graph(edge_index=torch.LongTensor(batch[key]).transpose(0, 1))
                    embs = self.model(G, return_dict=True)
                all_embs[key] = embs
        return all_embs

    def test_step(self, batch):
        all_embs, test_data = batch
        return evaluate_multiplex(all_embs, test_data)
