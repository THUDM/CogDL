import hashlib

import torch
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os

from .. import BaseModel


class Graph2Vec(BaseModel):
    r"""The Graph2Vec model from the `"graph2vec: Learning Distributed Representations of Graphs"
    <https://arxiv.org/abs/1707.05005>`_ paper

    Args:
        hidden_size (int) : The dimension of node representation.
        min_count (int) : Parameter in doc2vec.
        window_size (int) : The actual context size which is considered in language model.
        sampling_rate (float) : Parameter in doc2vec.
        dm (int) :  Parameter in doc2vec.
        iteration (int) : The number of iteration in WL method.
        lr (float) : Learning rate in doc2vec.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--window-size", type=int, default=0)
        parser.add_argument("--min-count", type=int, default=5)
        parser.add_argument("--dm", type=int, default=0)
        parser.add_argument("--sampling", type=float, default=0.0001)
        parser.add_argument("--iteration", type=int, default=2)
        parser.add_argument("--lr", type=float, default=0.025)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.min_count,
            args.window_size,
            args.sampling,
            args.dm,
            args.iteration,
            args.epochs,
            args.lr,
        )

    @staticmethod
    def feature_extractor(data, rounds, name):
        edge_index = torch.stack(data.edge_index)
        graph = nx.from_edgelist(np.array(edge_index.T.cpu(), dtype=int))
        if data.x is not None:
            feature = {int(key): str(val) for key, val in enumerate(np.array(data.x.cpu()))}
        else:
            feature = dict(nx.degree(graph))
        graph_wl_features = Graph2Vec.wl_iterations(graph, feature, rounds)
        doc = TaggedDocument(words=graph_wl_features, tags=["g_" + name])
        return doc

    @staticmethod
    def wl_iterations(graph, features, rounds):
        # TODO: solve hash and number
        nodes = graph.nodes
        wl_features = [str(val) for _, val in features.items()]
        for i in range(rounds):
            new_feats = {}
            for node in nodes:
                neighbors = graph.neighbors(node)
                neigh_feats = [features[x] for x in neighbors]
                neigh_feats = [features[node]] + sorted(neigh_feats)
                hash_feat = hashlib.md5("_".join([str(x) for x in neigh_feats]).encode())
                hash_feat = hash_feat.hexdigest()
                new_feats[node] = hash_feat
            wl_features = wl_features + list(new_feats.values())
            features = new_feats
        return wl_features

    def __init__(self, dimension, min_count, window_size, dm, sampling_rate, rounds, epochs, lr, worker=4):
        super(Graph2Vec, self).__init__()
        self.dimension = dimension
        self.min_count = min_count
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.dm = dm
        self.worker = worker
        self.rounds = rounds
        self.model = None
        self.doc_collections = None
        self.epochs = epochs
        self.lr = lr

    def forward(self, graphs, **kwargs):
        if self.doc_collections is None:
            self.doc_collections = Parallel(n_jobs=self.worker)(
                delayed(Graph2Vec.feature_extractor)(graph, self.rounds, str(i)) for i, graph in enumerate(graphs)
            )
        self.model = Doc2Vec(
            self.doc_collections,
            vector_size=self.dimension,
            window=self.window_size,
            min_count=self.min_count,
            dm=self.dm,
            sample=self.sampling_rate,
            workers=self.worker,
            epochs=self.epochs,
            alpha=self.lr,
        )
        vectors = np.array([self.model["g_" + str(i)] for i in range(len(graphs))])
        return vectors

    def save_embedding(self, output_path):
        self.model.wv.save(os.path.join(output_path, "model.wv"))
        self.model.wv.save_word2vec_format(os.path.join("model.emb"))
