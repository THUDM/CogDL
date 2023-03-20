import numpy as np
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
import random
from .. import BaseModel


class Metapath2vec(BaseModel):
    r"""The Metapath2vec model from the `"metapath2vec: Scalable Representation
    Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/
    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper

    Args:
        hidden_size (int) : The dimension of node representation.
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        window_size (int) : The actual context size which is considered in language model.
        worker (int) : The number of workers for word2vec.
        iteration (int) : The number of training iteration in word2vec.
        schema (str) : The metapath schema used in model. Metapaths are splited with ",",
        while each node type are connected with "-" in each metapath. For example:"0-1-0,0-2-0,1-0-2-0-1".
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
        parser.add_argument('--walk-num', type=int, default=20,
                            help='Number of walks per source. Default is 20.')
        parser.add_argument('--window-size', type=int, default=5,
                            help='Window size of skip-gram model. Default is 5.')
        parser.add_argument('--worker', type=int, default=10,
                            help='Number of parallel workers. Default is 10.')
        parser.add_argument('--iteration', type=int, default=10,
                            help='Number of iterations. Default is 10.')
        parser.add_argument('--schema', type=str, default="No",
                            help="Input schema for multi-type node representation.")
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
            args.schema,
        )

    def __init__(self, dimension, walk_length, walk_num, window_size, worker, iteration, schema):
        super(Metapath2vec, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration
        self.schema = schema
        self.node_type = None

    def forward(self, data):
        G = nx.DiGraph()
        row, col = data.edge_index
        G.add_edges_from(list(zip(row.numpy(), col.numpy())))
        self.G = G
        self.node_type = [str(a) for a in data.pos.tolist()]
        walks = self._simulate_walks(self.walk_length, self.walk_num, self.schema)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(
            walks,
            vector_size=self.dimension,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.worker,
            epochs=self.iteration,
        )
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([model.wv[str(id2node[i])] for i in range(len(id2node))])
        return embeddings

    def _walk(self, start_node, walk_length, schema=None):
        # Simulate a random walk starting from start node.
        # Note that metapaths in schema should be like '0-1-0', '0-2-0' or '1-0-2-0-1'.
        if schema:
            schema_items = schema.split("-")
            assert schema_items[0] == schema_items[-1]

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in list(self.G.neighbors(cur)):
                if schema is None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(random.choice(candidates))
            else:
                break
        # print(walk)
        return walk

    def _simulate_walks(self, walk_length, num_walks, schema="No"):
        # Repeatedly simulate random walks from each node with metapath schema.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if schema != "No":
            schema_list = schema.split(",")
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            print(str(walk_iter + 1), "/", str(num_walks))
            for node in nodes:
                if schema == "No":
                    walks.append(self._walk(node, walk_length))
                else:
                    for schema_iter in schema_list:
                        if schema_iter.split("-")[0] == self.node_type[node]:
                            walks.append(self._walk(node, walk_length, schema_iter))
        return walks
