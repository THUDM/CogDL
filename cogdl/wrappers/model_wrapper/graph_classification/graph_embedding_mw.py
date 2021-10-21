from torch.utils.data import DataLoader

from cogdl.data import MultiGraphDataset
from .. import EmbeddingModelWrapper
from cogdl.wrappers.tools.wrapper_utils import evaluate_graph_embeddings_using_svm


class GraphEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, model):
        super(GraphEmbeddingModelWrapper, self).__init__()
        self.model = model

    def train_step(self, batch):
        if isinstance(batch, DataLoader) or isinstance(batch, MultiGraphDataset):
            graphs = [x for x in batch]
        else:
            graphs = batch
        emb = self.model(graphs)
        return emb

    def test_step(self, batch):
        x, y = batch
        return evaluate_graph_embeddings_using_svm(x, y)
