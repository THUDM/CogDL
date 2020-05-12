import hashlib
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from .. import BaseModel, register_model

class WLMachine:
    def __init__(self, graph, features, rounds):
        self.rounds = rounds
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes
        self.wl_features = [str(val) for _, val in features]
        self.wl_iterations()

    def wl_iterations(self):
        for i in self.rounds:
            new_feats = {}
            for node in self.nodes:
                neighbors = self.graph.neighbors(node)
                neigh_feats = [self.features[x] for x in neighbors]
                neigh_feats = [self.features[node]] + sorted(neigh_feats)
                hash_feat = hashlib.md5("_".join(neigh_feats).encode())
                hash_feat = hash_feat.hexdigest()
                new_feats[node] = hash_feat
            self.wl_features = self.wl_features + list(new_feats.values())
            self.features = new_feats

    def get_wl_features(self):
        return self.wl_features



@register_model("graph2vec")
class Graph2Vec(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--window", type=int, default=0)
        parser.add_argument("--min-count", type=int, default=5)
        parser.add_argument("--dm", type=int, default=0)
        parser.add_argument("--sampling", type=float, default=0.0001)
        parser.add_argument("--iteration", type=int, default=2)
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--nn", type=bool, default=False)


    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.min_count,
            args.window,
            args.sampling,
            args.dm,
            args.iteration,
            args.epochs,
            args.lr
        )

    @staticmethod
    def feature_extractor(data, rounds, name):
        graph = nx.from_edgelist(np.array(data.edge_index.T.cpu(), dtype=int))
        if data.x is not None:
            feature = {int(key): str(val) for key, val in enumerate(np.array(data.x.cpu()))}
        else:
            feature = dict(nx.degree(graph))
        graph_wl_features = Graph2Vec.wl_iterations(graph, feature, rounds)
        doc = TaggedDocument(words=graph_wl_features, tags=["g_" + name])
        return doc

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

    def __init__(self, hidden_dim, min_count, window, dm, sampling_rate, rounds, epochs, lr, n_workers=4):
        super(Graph2Vec, self).__init__()
        self.hidden_dim = hidden_dim
        self.min_count = min_count
        self.window = window
        self.sampling_rate = sampling_rate
        self.dm = dm
        self.n_workers = n_workers
        self.rounds = rounds
        self.model = None
        self.doc_collections = None
        self.epochs = epochs
        self.lr = lr

    def forward(self, graphs, **kwargs):
        if self.doc_collections is None:
            self.doc_collections = Parallel(n_jobs=self.n_workers)(
                delayed(Graph2Vec.feature_extractor)(graph, self.rounds, str(i)) for i, graph in enumerate(graphs)
            )
        self.model = Doc2Vec(
            self.doc_collections,
            vector_size=self.hidden_dim,
            window=self.window,
            min_count=self.min_count,
            dm=self.dm,
            sample=self.sampling_rate,
            workers=self.n_workers,
            epochs=self.epochs,
            alpha=self.lr
        )
        vectors = np.array([self.model["g_"+str(i)] for i in range(len(graphs))])
        return vectors, None

    def save_embedding(self, output_path):
        self.model.wv.save("model.wv")
        self.model.wv.save_word2vec_format("model.emb")
