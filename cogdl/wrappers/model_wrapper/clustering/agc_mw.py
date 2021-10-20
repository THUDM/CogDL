from cogdl.wrappers.tools.wrapper_utils import evaluate_clustering
from .. import EmbeddingModelWrapper


class AGCModelWrapper(EmbeddingModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-clusters", type=int, default=7)
        parser.add_argument("--cluster-method", type=str, default="kmeans", help="option: kmeans or spectral")
        parser.add_argument("--evaluation", type=str, default="full", help="option: full or NMI")
        # fmt: on

    def __init__(self, model, optimizer_cfg, num_clusters, cluster_method="kmeans", evaluation="full", max_iter=5):
        super(AGCModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        self.full = evaluation == "full"

    def train_step(self, graph):
        emb = self.model.forward(graph)
        return emb

    def test_step(self, batch):
        features_matrix, graph = batch
        return evaluate_clustering(
            features_matrix, graph.y, self.cluster_method, self.num_clusters, graph.num_nodes, self.full
        )
