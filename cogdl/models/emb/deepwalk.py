import numpy as np
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
import random
from .. import BaseModel, register_model


@register_model("deepwalk")
class DeepWalk(BaseModel):
    r"""The DeepWalk model from the `"DeepWalk: Online Learning of Social Representations"
    <https://arxiv.org/abs/1403.6652>`_ paper
    
    Args:
        hidden_size (int) : The dimension of node representation.
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        window_size (int) : The actual context size which is considered in language model.
        worker (int) : The number of workers for word2vec.
        iteration (int) : The number of training iteration in word2vec.
    """
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
        parser.add_argument('--walk-num', type=int, default=40,
                            help='Number of walks per source. Default is 40.')
        parser.add_argument('--window-size', type=int, default=5,
                            help='Window size of skip-gram model. Default is 5.')
        parser.add_argument('--worker', type=int, default=10,
                            help='Number of parallel workers. Default is 10.')
        parser.add_argument('--iteration', type=int, default=10,
                            help='Number of iterations. Default is 10.')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.window_size,
            args.worker,
            args.iteration,
        )

    def __init__(
        self, dimension, walk_length, walk_num, window_size, worker, iteration
    ):
        super(DeepWalk, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration

    def train(self, G):
        self.G = G
        walks = self._simulate_walks(self.walk_length, self.walk_num)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(
            walks,
            size=self.dimension,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.worker,
            iter=self.iteration,
        )
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([model.wv[str(id2node[i])] for i in range(len(id2node))])
        return embeddings

    def _walk(self, start_node, walk_length):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            k = int(np.floor(np.random.rand() * len(cur_nbrs)))
            walk.append(cur_nbrs[k])
        return walk

    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("node number:", len(nodes))
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, walk_length))
        return walks
