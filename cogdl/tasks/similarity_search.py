import argparse
import networkx as nx
import numpy as np
import torch
from collections import defaultdict

from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task


@register_task("similarity_search")
class SimilaritySearch(BaseTask):
    """Similarity Search task."""

    @staticmethod
    def add_args(_: argparse.ArgumentParser):
        """Add task-specific arguments to the parser."""
        # need no extra argument
        pass

    def __init__(self, args, dataset=None, model=None):
        super(SimilaritySearch, self).__init__(args)
        dataset = build_dataset(args) if dataset is None else dataset
        self.data = dataset.data
        model = build_model(args) if model is None else model
        self.model = model
        self.hidden_size = args.hidden_size
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]

    def _evaluate(self, emb_1, emb_2, dict_1, dict_2):
        shared_keys = set(dict_1.keys()) & set(dict_2.keys())
        shared_keys = list(
            filter(
                lambda x: dict_1[x] < emb_1.shape[0] and dict_2[x] < emb_2.shape[0],
                shared_keys,
            )
        )
        emb_1 /= np.linalg.norm(emb_1, axis=1).reshape(-1, 1)
        emb_2 /= np.linalg.norm(emb_2, axis=1).reshape(-1, 1)
        reindex = [dict_2[key] for key in shared_keys]
        reindex_dict = dict([(x, i) for i, x in enumerate(reindex)])
        emb_2 = emb_2[reindex]
        k_list = [20, 40]
        # id2name = dict([(dict_2[k], k) for k in dict_2])

        all_results = defaultdict(list)
        for key in shared_keys:
            v = emb_1[dict_1[key]]
            scores = emb_2.dot(v)

            idxs = scores.argsort()[::-1]
            for k in k_list:
                all_results[k].append(int(reindex_dict[dict_2[key]] in idxs[:k]))
        res = dict((f"Recall @ {k}", sum(all_results[k]) / len(all_results[k])) for k in k_list)

        return res

    def _train_wrap(self, data):
        G = nx.MultiGraph()
        row, col = data.edge_index
        row, col = row.numpy(), col.numpy()
        G.add_edges_from(list(zip(row, col)))
        embeddings = self.model.train(data)
        # Map node2id
        features_matrix = np.zeros((G.number_of_nodes(), self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]
        return features_matrix

    def train(self):
        emb_1 = self._train_wrap(self.data[0])
        emb_2 = self._train_wrap(self.data[1])
        return dict(self._evaluate(emb_1, emb_2, self.data[0].y, self.data[1].y))
