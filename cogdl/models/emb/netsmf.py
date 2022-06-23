import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from multiprocessing import Pool
from tqdm import tqdm
import time

from cogdl.utils import alias_draw, alias_setup
from .. import BaseModel


class NetSMF(BaseModel):
    r"""The NetSMF model from the `"NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization"
    <http://arxiv.org/abs/1710.02971>`_ paper.

    Args:
        hidden_size (int) : The dimension of node representation.
        window_size (int) : The actual context size which is considered in language model.
        negative (int) : The number of nagative samples in negative sampling.
        num_round (int) : The number of round in NetSMF.
        worker (int) : The number of workers for NetSMF.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--window-size", type=int, default=10,
                            help="Window size of approximate matrix. Default is 10.")
        parser.add_argument("--negative", type=int, default=1,
                            help="Number of negative node in sampling. Default is 1.")
        parser.add_argument("--num-round", type=int, default=100,
                            help="Number of round in NetSMF. Default is 100.")
        parser.add_argument("--worker", type=int, default=10,
                            help="Number of parallel workers. Default is 10.")
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.window_size, args.negative, args.num_round, args.worker,)

    def __init__(self, dimension, window_size, negative, num_round, worker):
        super(NetSMF, self).__init__()
        self.dimension = dimension
        self.window_size = window_size
        self.negative = negative
        self.worker = worker
        self.num_round = num_round

    def forward(self, graph, return_dict=False):
        self.G = graph.to_networkx()
        node2id = dict([(node, vid) for vid, node in enumerate(self.G.nodes())])
        self.is_directed = nx.is_directed(self.G)
        self.num_node = self.G.number_of_nodes()
        self.num_edge = self.G.number_of_edges()
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]

        id2node = dict(zip(node2id.values(), node2id.keys()))

        self.num_neigh = np.asarray([len(list(self.G.neighbors(id2node[i]))) for i in range(self.num_node)])
        self.neighbors = [[node2id[v] for v in self.G.neighbors(id2node[i])] for i in range(self.num_node)]
        s = time.time()
        self.alias_nodes = {}
        self.node_weight = {}
        for i in range(self.num_node):
            unnormalized_probs = [self.G[id2node[i]][nbr].get("weight", 1.0) for nbr in self.G.neighbors(id2node[i])]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[i] = alias_setup(normalized_probs)
            self.node_weight[i] = dict(zip([node2id[nbr] for nbr in self.G.neighbors(id2node[i])], unnormalized_probs,))

        t = time.time()
        print("alias_nodes", t - s)

        # run netsmf algorithm with multiprocessing and apply randomized svd
        print("number of sample edges ", self.num_round * self.num_edge * self.window_size)
        print("random walk start...")
        t0 = time.time()
        results = []
        pool = Pool(processes=self.worker)
        for i in range(self.worker):
            results.append(pool.apply_async(func=self._random_walk_matrix, args=(i,)))
        pool.close()
        pool.join()
        print("random walk time", time.time() - t0)

        matrix = sp.csr_matrix((self.num_node, self.num_node))
        A = sp.csr_matrix(nx.adjacency_matrix(self.G))
        degree = sp.diags(np.array(A.sum(axis=0))[0], format="csr")
        degree_inv = degree.power(-1)

        t1 = time.time()
        for res in results:
            matrix += res.get()
        t2 = time.time()
        print("construct random walk matrix time", time.time() - t1)

        L = sp.csgraph.laplacian(matrix, normed=False, return_diag=False)
        M = degree_inv.dot(degree - L).dot(degree_inv)
        M = M * A.sum() / self.negative
        M.data[M.data <= 1] = 1
        M.data = np.log(M.data)
        M.eliminate_zeros()
        print("number of nzz", M.nnz)
        print("construct matrix sparsifier time", time.time() - t2)

        embeddings = self._get_embedding_rand(M)

        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(self.G.nodes()):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((graph.num_nodes, embeddings.shape[1]))
            nx_nodes = self.G.nodes()
            features_matrix[nx_nodes] = embeddings[np.arange(graph.num_nodes)]
        return features_matrix

    def _get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]  # noqa E741
        smat = sp.csc_matrix(matrix)
        print("svd sparse", smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print("sparsesvd time", time.time() - t1)
        return U

    def _path_sampling(self, u, v, r):
        # sample a r-length path from edge(u, v) and return path end node
        k = np.random.randint(r) + 1
        zp, rand_u, rand_v = 2.0 / self.node_weight[u][v], k - 1, r - k
        for i in range(rand_u):
            new_u = self.neighbors[u][alias_draw(self.alias_nodes[u][0], self.alias_nodes[u][1])]
            zp += 2.0 / self.node_weight[u][new_u]
            u = new_u
        for j in range(rand_v):
            new_v = self.neighbors[v][alias_draw(self.alias_nodes[v][0], self.alias_nodes[v][1])]
            zp += 2.0 / self.node_weight[v][new_v]
            v = new_v
        return u, v, zp

    def _random_walk_matrix(self, pid):
        # construct matrix based on random walk
        np.random.seed(pid)
        matrix = sp.lil_matrix((self.num_node, self.num_node))
        for i in tqdm(range(self.num_edge * self.num_round // self.worker)):
            u, v = self.edges[i % self.num_edge]
            if not self.is_directed and np.random.rand() > 0.5:
                v, u = u, v
            for r in range(1, self.window_size + 1):
                u_, v_, zp = self._path_sampling(u, v, r)
                matrix[u_, v_] += 2 * r / self.window_size / self.num_round / zp
        return matrix.tocsr()
