from .. import register_data_wrapper, DataWrapper
from cogdl.wrappers.wrapper_utils import node_degree_as_feature
from cogdl.data import DataLoader


@register_data_wrapper("graph_classification_dw")
class GraphClassificationDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--degree-node-features", action="store_true",
                            help="Use one-hot degree vector as input node features")
        # parser.add_argument("--kfold", action="store_true", help="Use 10-fold cross-validation")
        parser.add_argument("--train-ratio", type=float, default=0.5)
        parser.add_argument("--test-ratio", type=float, default=0.3)
        # fmt: on

    def __init__(self, dataset, degree_node_features=False, batch_size=32, train_ratio=0.5, test_ratio=0.3):
        super(GraphClassificationDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.degree_node_features = degree_node_features
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size

    def training_wrapper(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_wrapper(self):
        pass

    def pre_transform(self):
        if self.degree_node_features:
            self.dataset = node_degree_as_feature(self.dataset)
