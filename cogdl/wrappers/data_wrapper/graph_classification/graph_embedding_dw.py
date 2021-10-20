import numpy as np

from .. import DataWrapper
from cogdl.wrappers.tools.wrapper_utils import node_degree_as_feature


class GraphEmbeddingDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--degree-node-features", action="store_true",
                            help="Use one-hot degree vector as input node features")
        # fmt: on

    def __init__(self, dataset, degree_node_features=False):
        super(GraphEmbeddingDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.degree_node_features = degree_node_features

    def train_wrapper(self):
        return self.dataset

    def test_wrapper(self):
        if self.dataset[0].y.shape[0] > 1:
            return np.array([g.y.numpy() for g in self.dataset])
        else:
            return np.array([g.y.item() for g in self.dataset])

    def pre_transform(self):
        if self.degree_node_features:
            self.dataset = node_degree_as_feature(self.dataset)
