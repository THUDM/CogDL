from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
import numpy as np

from torch.utils.data import DataLoader
from .. import register_model_wrapper, EmbeddingModelWrapper


@register_model_wrapper("graph_embedding_mw")
class GraphEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self):
        super(GraphEmbeddingModelWrapper, self).__init__()

    def train_step(self, batch):
        if isinstance(batch, DataLoader):
            graphs = [x for x in batch]
        else:
            graphs = batch
        emb = self.model(graphs)
        return emb

    def test_step(self, batch):
        x, y = batch
        return evaluate_graph_embeddings(x, y)


def evaluate_graph_embeddings(embeddings, labels):
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

    return dict(Acc=test_f1, Std=test_std)
