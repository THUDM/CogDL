from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_liblinear
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
        return evaluate_node_embeddings_using_liblinear(x, y, self.num_shuffle, self.training_percents)
