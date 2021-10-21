import scipy.sparse as sp
import networkx as nx

from cogdl.models import build_model
from cogdl.utils.utils import ArgClass
from cogdl.wrappers.tools.wrapper_utils import evaluate_node_embeddings_using_liblinear
from .. import EmbeddingModelWrapper


class NetworkEmbeddingModelWrapper(EmbeddingModelWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--num-shuffle", type=int, default=10)
        parser.add_argument("--training-percents", default=[0.9], type=float, nargs="+")
        parser.add_argument("--enhance", type=str, default=None, help="use prone or prone++ to enhance embedding")
        parser.add_argument("--max-evals", type=int, default=10)
        parser.add_argument("--num-workers", type=int, default=1)
        # fmt: on

    def __init__(self, model, num_shuffle=1, training_percents=[0.1], enhance=None, max_evals=10, num_workers=1):
        super(NetworkEmbeddingModelWrapper, self).__init__()
        self.model = model
        self.num_shuffle = num_shuffle
        self.training_percents = training_percents
        self.enhance = enhance
        self.max_evals = max_evals
        self.num_workers = num_workers

    def train_step(self, batch):
        emb = self.model(batch)
        if self.enhance is not None:
            self._enhance_emb(batch, emb)
        return emb

    def test_step(self, batch):
        x, y = batch
        return evaluate_node_embeddings_using_liblinear(x, y, self.num_shuffle, self.training_percents)

    def _enhance_emb(self, graph, embs):
        A = nx.to_scipy_sparse_matrix(graph.to_networkx())
        args = ArgClass()
        if self.enhance == "prone":
            args.model = "prone"
            args.hidden_size = embs.shape[1]
            args.step, args.theta, args.mu = 5, 0.5, 0.2
            model = build_model(args)
            embs = model._chebyshev_gaussian(A, embs)
        elif self.enhance == "prone++":
            args.model = "prone++"
            args.filter_types = ["heat", "ppr", "gaussian", "sc"]
            args.max_evals = self.max_evals
            args.num_workers = self.num_workers
            args.no_svd = False
            args.loss = "infomax"
            args.no_search = False
            model = build_model(args)
            embs = model(embs, A)
        else:
            raise ValueError("only supports 'prone' and 'prone++'")
        return embs
