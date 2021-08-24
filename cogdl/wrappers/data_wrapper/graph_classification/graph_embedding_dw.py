from torch.utils.data import DataLoader

from .. import register_data_wrapper, DataWrapper
from cogdl.wrappers.wrapper_utils import node_degree_as_feature


@register_data_wrapper("graph_embedding_dw")
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

    def training_wrapper(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=False)

    def test_wrapper(self):
        return self.dataset, [g.y for g in self.dataset]

    def pre_transform(self):
        if self.degree_node_features:
            self.dataset = node_degree_as_feature(self.dataset)
