import time

import networkx as nx
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from cogdl.utils import alias_draw, alias_setup
from .. import BaseModel


class PTE(BaseModel):
    r"""The PTE model from the `"PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks"
    <https://arxiv.org/abs/1508.00200>`_ paper.

    Args:
        hidden_size (int) : The dimension of node representation.
        walk_length (int) : The walk length.
        walk_num (int) : The number of walks to sample for each node.
        negative (int) : The number of nagative samples for each edge.
        batch_size (int) : The batch size of training in PTE.
        alpha (float) : The initial learning rate of SGD.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
        parser.add_argument('--walk-num', type=int, default=20,
                            help='Number of walks per source. Default is 20.')
        parser.add_argument('--negative', type=int, default=5,
                            help='Number of negative node in sampling. Default is 5.')
        parser.add_argument('--batch-size', type=int, default=1000,
                            help='Batch size in SGD training process. Default is 1000.')
        parser.add_argument('--alpha', type=float, default=0.025,
                            help='Initial learning rate of SGD. Default is 0.025.')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.walk_length, args.walk_num, args.negative, args.batch_size, args.alpha,)

    def __init__(self, dimension, walk_length, walk_num, negative, batch_size, alpha):
        super(PTE, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.negative = negative
        self.batch_size = batch_size
        self.init_alpha = alpha

    def forward(self, data):
        G = nx.DiGraph()
        row, col = data.edge_index
        G.add_edges_from(list(zip(row.numpy(), col.numpy())))
        self.G = G
        self.node_type = data.pos.tolist()
        self.num_node = G.number_of_nodes()
        self.num_edge = G.number_of_edges()
        self.num_sampling_edge = self.walk_length * self.walk_num * self.num_node

        self.num_node_type = len(set(self.node_type))
        context_node = [nid for nid, ntype in enumerate(self.node_type) if ntype == 0]

        self.edges, self.edges_prob = [[] for _ in range(self.num_node_type)], []
        self.node_prob, self.id2node = [], [dict() for _ in range(self.num_node_type)]

        subgraphs = []
        for i in range(self.num_node_type):
            for j in range(i + 1, self.num_node_type):
                context_node = [nid for nid, ntype in enumerate(self.node_type) if ntype == i or ntype == j]
                sub_graph = nx.Graph()
                sub_graph = self.G.subgraph(context_node)
                if sub_graph.number_of_edges() != 0:
                    subgraphs.append(sub_graph)
        self.num_graph = len(subgraphs)
        print("number of subgraph", self.num_graph)

        for i in range(self.num_graph):
            self.edges[i] = [[e[0], e[1]] for e in subgraphs[i].edges()]
            edges_prob = np.asarray([subgraphs[i][u][v].get("weight", 1.0) for u, v in self.edges[i]])
            edges_prob /= np.sum(edges_prob)
            edges_table_prob = alias_setup(edges_prob)
            self.edges_prob.append(edges_table_prob)

            context_node = subgraphs[i].nodes()
            self.id2node[i] = dict(zip(range(len(context_node)), context_node))
            node2id = dict(zip(context_node, range(len(context_node))))

            degree_weight = np.asarray([0] * len(context_node))
            for u in context_node:
                for v in list(subgraphs[i].neighbors(u)):
                    degree_weight[node2id[u]] += subgraphs[i][u][v].get("weight", 1.0)

            node_prob = np.power(degree_weight, 0.75)
            node_prob /= np.sum(node_prob)
            nodes_table_prob = alias_setup(node_prob)
            self.node_prob.append(nodes_table_prob)

        print("train pte with 2-order")
        self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
        self.emb_context = self.emb_vertex
        self._train_line()
        embedding = preprocessing.normalize(self.emb_vertex, "l2")
        return embedding

    def _update(self, vec_u, vec_v, vec_error, label):
        # update vetex embedding and vec_error
        f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
        g = (self.alpha * (label - f)).reshape((len(label), 1))
        vec_error += g * vec_v
        vec_v += g * vec_u

    def _train_line(self,):
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

            for k in range(self.num_graph):
                u, v = [0] * batch_size, [0] * batch_size
                for i in range(batch_size):
                    edge_id = alias_draw(self.edges_prob[k][0], self.edges_prob[k][1])
                    u[i], v[i] = self.edges[k][edge_id]

                vec_error = np.zeros((batch_size, self.dimension))
                label, target = np.asarray([1 for i in range(batch_size)]), np.asarray(v)
                for j in range(1 + self.negative):
                    if j != 0:
                        label = np.asarray([0 for i in range(batch_size)])
                        for i in range(batch_size):
                            neg_node = alias_draw(self.node_prob[k][0], self.node_prob[k][1])
                            target[i] = self.id2node[k][neg_node]
                        self._update(self.emb_vertex[u], self.emb_context[target], vec_error, label)
                self.emb_vertex[u] += vec_error
