import hashlib
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from .. import BaseModel, register_model


@register_model("dgk")
class DeepGraphKernel(BaseModel):
    r"""The Hin2vec model from the `"Deep Graph Kernels"
    <https://dl.acm.org/citation.cfm?id=2783417&CFID=763322570&CFTOKEN=93890155>`_ paper.
    
    Args:
        hidden_size (int) : The dimension of node representation.
        min_count (int) : Parameter in word2vec.
        window (int) : The actual context size which is considered in language model.
        sampling_rate (float) : Parameter in word2vec.
        iteration (int) : The number of iteration in WL method.
        epoch (int) : The number of training iteration.
        alpha (float) : The learning rate of word2vec.
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--window_size", type=int, default=5)
        parser.add_argument("--min-count", type=int, default=1)
        parser.add_argument("--sampling", type=float, default=0.0001)
        parser.add_argument("--iteration", type=int, default=2)
        parser.add_argument("--epoch", type=int, default=20)
        parser.add_argument("--alpha", type=float, default=0.01)


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.min_count,
            args.window_size,
            args.sampling,
            args.iteration,
            args.epoch,
            args.alpha
        )

    @staticmethod
    def feature_extractor(data, rounds, name):
        graph = nx.from_edgelist(np.array(data.edge_index.T.cpu(), dtype=int))
        if data.x is not None:
            feature = {int(key): str(val.argmax(axis=0)) for key, val in enumerate(np.array(data.x.cpu()))}
        else:
            feature = dict(nx.degree(graph))
        graph_wl_features = DeepGraphKernel.wl_iterations(graph, feature, rounds)
        return graph_wl_features

    @staticmethod
    def wl_iterations(graph, features, rounds):
        #TODO: solve hash and number
        nodes = graph.nodes
        wl_features = [str(val) for _, val in features.items()]
        for i in range(rounds):
            new_feats = {}
            for node in nodes:
                neighbors = graph.neighbors(node)
                neigh_feats = [features[x] for x in neighbors]
                neigh_feats = [features[node]] + sorted(neigh_feats)
                hash_feat = hashlib.md5("_".join(neigh_feats).encode())
                hash_feat = hash_feat.hexdigest()
                new_feats[node] = hash_feat
            wl_features = wl_features + list(new_feats.values())
            features = new_feats
        return wl_features

    def __init__(self, hidden_dim, min_count, window_size, sampling_rate, rounds, epoch, alpha, n_workers=4):
        super(DeepGraphKernel, self).__init__()
        self.hidden_dim = hidden_dim
        self.min_count = min_count
        self.window = window_size
        self.sampling_rate = sampling_rate
        self.n_workers = n_workers
        self.rounds = rounds
        self.model = None
        self.gl_collections = None
        self.epoch = epoch
        self.alpha = alpha

    def forward(self, graphs, **kwargs):
        if self.gl_collections is None:
            self.gl_collections = Parallel(n_jobs=self.n_workers)(
                delayed(DeepGraphKernel.feature_extractor)(graph, self.rounds, str(i)) for i, graph in enumerate(graphs)
            )
        
        model = Word2Vec(
            self.gl_collections,
            size=self.hidden_dim,
            window=self.window,
            min_count=self.min_count,
            sample=self.sampling_rate,
            workers=self.n_workers,
            iter=self.epoch,
            alpha=self.alpha
        )
        vectors = np.asarray([model.wv[str(node)] for node in model.wv.index2word])
        S = vectors.dot(vectors.T)        
        node2id = dict(zip(model.wv.index2word, range(len(model.wv.index2word))))
        
        num_graph, size_vocab = len(graphs), len(node2id)
        norm_prob = np.zeros((num_graph, size_vocab))
        for i, gls in enumerate(self.gl_collections):
            for gl in gls:
                if gl in node2id:
                    norm_prob[i, node2id[gl]] += 1
            # norm_prob[i] /= sum(norm_prob[i])
        embedding = norm_prob.dot(S)
        return embedding, None

    def save_embedding(self, output_path):
        self.model.wv.save("model.wv")
        self.model.wv.save_word2vec_format("model.emb")
