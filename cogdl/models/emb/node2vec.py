import numpy as np
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
import random
import time
from .. import BaseModel, register_model, alias_draw, alias_setup


@register_model("node2vec")
class Node2vec(BaseModel):
    r"""The node2vec model from the `"node2vec: Scalable feature learning for networks"
    <http://dl.acm.org/citation.cfm?doid=2939672.2939754>`_ paper
    
    Args:
        hidden_size (int) : The dimension of node representation.
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        window_size (int) : The actual context size which is considered in language model.
        worker (int) : The number of workers for word2vec.
        iteration (int) : The number of training iteration in word2vec.
        p (float) : Parameter in node2vec.
        q (float) : Parameter in node2vec.
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
        parser.add_argument('--p', type=float, default=1.0,
                            help='Parameter in node2vec. Default is 1.0.')
        parser.add_argument('--q', type=float, default=1.0,
                            help='Parameter in node2vec. Default is 1.0.')
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
            args.p,
            args.q,
        )

    def __init__(
        self, dimension, walk_length, walk_num, window_size, worker, iteration, p, q
    ):
        super(Node2vec, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration
        self.p = p
        self.q = q

    def train(self, G):
        self.G = G
        is_directed = nx.is_directed(self.G)
        for i, j in G.edges():
            G[i][j]["weight"] = G[i][j].get("weight", 1.0)
            if not is_directed:
                G[j][i]["weight"] = G[j][i].get("weight", 1.0)
        self._preprocess_transition_probs()
        walks = self._simulate_walks(self.walk_num, self.walk_length)
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
        self.embeddings = np.asarray(
            [model.wv[str(id2node[i])] for i in range(len(id2node))]
        )
        return self.embeddings

    def _node2vec_walk(self, walk_length, start_node):
        # Simulate a random walk starting from start node.
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    next = cur_nbrs[
                        alias_draw(
                            alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1]
                        )
                    ]
                    walk.append(next)
            else:
                break

        return walk

    def _simulate_walks(self, num_walks, walk_length):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            if walk_iter % 10 == 0:
                print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(
                    self._node2vec_walk(walk_length=walk_length, start_node=node)
                )

        return walks

    def _get_alias_edge(self, src, dst):
        # Get the alias edge setup lists for a given edge.
        G = self.G
        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / self.p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def _preprocess_transition_probs(self):
        # Preprocessing of transition probabilities for guiding the random walks.
        G = self.G
        is_directed = nx.is_directed(self.G)

        print(len(list(G.nodes())))
        print(len(list(G.edges())))

        s = time.time()
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = alias_setup(normalized_probs)

        t = time.time()
        print("alias_nodes", t - s)

        alias_edges = {}
        s = time.time()

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])

        t = time.time()
        print("alias_edges", t - s)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return
