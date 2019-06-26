import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from multiprocessing import Pool
import time

from . import BaseModel, register_model


@register_model("netsmf")
class NetSMF(BaseModel):
	@staticmethod
	def add_args(parser):
		"""Add model-specific arguments to the parser."""
		# fmt: off
		parser.add_argument('--window-size', type=int, default=5,
		                    help='Window size of skip-gram model. Default is 5.')
		parser.add_argument('--negative', type=int, default=5,
		                    help='Number of negative node in sampling. Default is 5.')
		parser.add_argument('--num-round', type=int, default=100,
		                    help='Number of round in NetSMF. Default is 100.')
		parser.add_argument('--worker', type=int, default=10,
		                    help='Number of parallel workers. Default is 10.')
		# fmt: on

	@classmethod
	def build_model_from_args(cls, args):
		return cls(args.hidden_size, args.window_size, args.negative, args.num_round, args.worker)

	def __init__(self, dimension, window_size, negative, num_round, worker):
		super(NetSMF, self).__init__()
		self.dimension = dimension
		self.window_size = window_size
		self.negative = negative
		self.worker = worker
		self.num_round = num_round

	def train(self, G):
		self.G = G
		node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])
		self.is_directed = nx.is_directed(self.G)
		self.num_node = self.G.number_of_nodes()
		self.num_edge = G.number_of_edges()
		self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]

		id2node = dict(zip(node2id.values(), node2id.keys()))

		self.num_neigh = np.asarray([len(list(self.G.neighbors(id2node[i]))) for i in range(self.num_node)])
		self.neighbors = [[node2id[v] for v in self.G.neighbors(id2node[i])]
		                  for i in range(self.num_node)]

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
		print('random walk time', time.time() - t0)

		matrix = sp.lil_matrix((self.num_node, self.num_node))
		A = sp.csr_matrix(nx.adjacency_matrix(self.G))
		degree = sp.diags(np.array(A.sum(axis=0))[0], format="csr")
		degree_inv = degree.power(-1)

		t1 = time.time()
		for res in results:
			matrix += res.get()
		print('number of nzz', matrix.nnz)
		t2 = time.time()
		print('construct random walk matrix time', time.time() - t1)

		L = sp.csgraph.laplacian(matrix, normed=False, return_diag=False)
		M = degree_inv.dot(degree - L).dot(degree_inv)
		M = M * A.sum() / self.negative
		M.data[M.data <= 1] = 1
		M.data = np.log(M.data)
		print('construct matrix sparsifier time', time.time() - t2)

		embedding = self._get_embedding_rand(M)
		return embedding


	def _get_embedding_rand(self, matrix):
		# Sparse randomized tSVD for fast embedding
		t1 = time.time()
		l = matrix.shape[0]
		smat = sp.csc_matrix(matrix)
		print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
		U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
		U = U * np.sqrt(Sigma)
		U = preprocessing.normalize(U, "l2")
		print('sparsesvd time', time.time() - t1)
		return U


	def _path_sampling(self, u, v, r):
		# sample a r-length path from edge(u, v) and return path end node
		k = np.random.randint(r) + 1
		rand_u, rand_v = k - 1, r - k
		for i in range(rand_u):
			u = self.neighbors[u][np.random.randint(self.num_neigh[u])]
		for j in range(rand_v):
			v = self.neighbors[v][np.random.randint(self.num_neigh[v])]
		return u, v

	def _random_walk_matrix(self, pid):
		# construct matrix based on random walk
		np.random.seed(pid)
		matrix = sp.lil_matrix((self.num_node, self.num_node))
		t0 = time.time()
		for round in range(int(self.num_round / self.worker)):
			if round % 10 == 0 and pid == 0:
				print("round %d / %d, time: %lf" % (round * self.worker, self.num_round, time.time() - t0))
			for i in range(self.num_edge):
				u, v = self.edges[i]
				if not self.is_directed and np.random.rand() > 0.5:
					v, u = self.edges[i]
				for r in range(1, self.window_size + 1):
					u_, v_ = self._path_sampling(u, v, r)
					matrix[u_, v_] += 1.0
		return matrix



