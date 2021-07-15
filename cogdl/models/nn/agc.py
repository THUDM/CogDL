from .. import BaseModel, register_model
from cogdl.trainers.agc_trainer import AGCTrainer


@register_model("agc")
class AGC(BaseModel):
    r"""The AGC model from the `"Attributed Graph Clustering via Adaptive Graph Convolution"
    <https://arxiv.org/abs/1906.01210>`_ paper

    Args:
        num_clusters (int) : Number of clusters.
        max_iter     (int) : Max iteration to increase k
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--max-iter", type=int, default=60)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_clusters, args.max_iter)

    def __init__(self, num_clusters, max_iter):
        super(AGC, self).__init__()

        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.k = 0
        self.features_matrix = None

    @staticmethod
    def get_trainer(args):
        return AGCTrainer

    def get_features(self, data):
        return self.features_matrix.detach().cpu()
