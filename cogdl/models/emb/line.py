import time

import networkx as nx
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from cogdl.utils import alias_draw, alias_setup
from .. import BaseModel


class LINE(BaseModel):
    r"""The LINE model from the `"Line: Large-scale information network embedding"
    <http://arxiv.org/abs/1503.03578>`_ paper.

    Args:
        hidden_size (int) : The dimension of node representation.
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        negative (int) : The number of nagative samples for each edge.
        batch_size (int) : The batch size of training in LINE.
        alpha (float) : The initial learning rate of SGD.
        order (int) : 1 represents perserving 1-st order proximity, 2 represents 2-nd,
        while 3 means both of them (each of them having dimension/2 node representation).
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--walk-length", type=int, default=80,
                            help="Length of walk per source. Default is 80.")
        parser.add_argument("--walk-num", type=int, default=20,
                            help="Number of walks per source. Default is 20.")
        parser.add_argument("--negative", type=int, default=5,
                            help="Number of negative node in sampling. Default is 5.")
        parser.add_argument("--batch-size", type=int, default=1000,
                            help="Batch size in SGD training process. Default is 1000.")
        parser.add_argument("--alpha", type=float, default=0.025,
                            help="Initial learning rate of SGD. Default is 0.025.")
        parser.add_argument("--order", type=int, default=3,
                            help="Order of proximity in LINE. Default is 3 for 1+2.")
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size, args.walk_length, args.walk_num, args.negative, args.batch_size, args.alpha, args.order,
        )

    def __init__(self, dimension, walk_length, walk_num, negative, batch_size, alpha, order):
        super(LINE, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.negative = negative
        self.batch_size = batch_size
        self.init_alpha = alpha
        self.order = order

    def forward(self, graph, return_dict=False):
        # run LINE algorithm, 1-order, 2-order or 3(1-order + 2-order)
        nx_g = graph.to_networkx()
        self.G = nx_g
        self.is_directed = nx.is_directed(self.G)
        self.num_node = nx_g.number_of_nodes()
        self.num_edge = nx_g.number_of_edges()
        self.num_sampling_edge = self.walk_length * self.walk_num * self.num_node

        node2id = dict([(node, vid) for vid, node in enumerate(nx_g.nodes())])
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]
        self.edges_prob = np.asarray([nx_g[u][v].get("weight", 1.0) for u, v in nx_g.edges()])
        self.edges_prob /= np.sum(self.edges_prob)
        self.edges_table, self.edges_prob = alias_setup(self.edges_prob)

        degree_weight = np.asarray([0] * self.num_node)
        for u, v in nx_g.edges():
            degree_weight[node2id[u]] += nx_g[u][v].get("weight", 1.0)
            if not self.is_directed:
                degree_weight[node2id[v]] += nx_g[u][v].get("weight", 1.0)
        self.node_prob = np.power(degree_weight, 0.75)
        self.node_prob /= np.sum(self.node_prob)
        self.node_table, self.node_prob = alias_setup(self.node_prob)

        if self.order == 3:
            self.dimension = int(self.dimension / 2)
        if self.order == 1 or self.order == 3:
            print("train line with 1-order")
            print(type(self.dimension))
            self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
            self._train_line(order=1)
            embedding1 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 2 or self.order == 3:
            print("train line with 2-order")
            self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
            self.emb_context = self.emb_vertex
            self._train_line(order=2)
            embedding2 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 1:
            embeddings = embedding1
        elif self.order == 2:
            embeddings = embedding2
        else:
            print("concatenate two embedding...")
            embeddings = np.hstack((embedding1, embedding2))

        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(nx_g.nodes()):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((graph.num_nodes, embeddings.shape[1]))
            nx_nodes = nx_g.nodes()
            features_matrix[nx_nodes] = embeddings[np.arange(graph.num_nodes)]
        return features_matrix

    def _update(self, vec_u, vec_v, vec_error, label):
        # update vetex embedding and vec_error
        f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
        g = (self.alpha * (label - f)).reshape((len(label), 1))
        vec_error += g * vec_v
        vec_v += g * vec_u

    def _train_line(self, order):
        # train Line model with order
        self.alpha = self.init_alpha
        batch_size = self.batch_size
        t0 = time.time()
        num_batch = int(self.num_sampling_edge / batch_size)
        epoch_iter = tqdm(range(num_batch))
        for b in epoch_iter:
            if b % 100 == 0:
                epoch_iter.set_description(
                    f"Progress: {b *1.0/num_batch * 100:.4f}%, alpha: {self.alpha:.6f}, time: {time.time() - t0:.4f}"
                )
                self.alpha = self.init_alpha * max((1 - b * 1.0 / num_batch), 0.0001)
            u, v = [0] * batch_size, [0] * batch_size
            for i in range(batch_size):
                edge_id = alias_draw(self.edges_table, self.edges_prob)
                u[i], v[i] = self.edges[edge_id]
                if not self.is_directed and np.random.rand() > 0.5:
                    v[i], u[i] = self.edges[edge_id]

            vec_error = np.zeros((batch_size, self.dimension))
            label, target = np.asarray([1 for i in range(batch_size)]), np.asarray(v)
            for j in range(1 + self.negative):
                if j != 0:
                    label = np.asarray([0 for i in range(batch_size)])
                    for i in range(batch_size):
                        target[i] = alias_draw(self.node_table, self.node_prob)
                if order == 1:
                    self._update(self.emb_vertex[u], self.emb_vertex[target], vec_error, label)
                else:
                    self._update(self.emb_vertex[u], self.emb_context[target], vec_error, label)
            self.emb_vertex[u] += vec_error
