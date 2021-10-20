from .. import DataWrapper
from cogdl.wrappers.tools.wrapper_utils import node_degree_as_feature, split_dataset
from cogdl.data import DataLoader


class GraphClassificationDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--degree-node-features", action="store_true",
                            help="Use one-hot degree vector as input node features")
        # parser.add_argument("--kfold", action="store_true", help="Use 10-fold cross-validation")
        parser.add_argument("--train-ratio", type=float, default=0.5)
        parser.add_argument("--test-ratio", type=float, default=0.3)
        parser.add_argument("--batch-size", type=int, default=16)
        # fmt: on

    def __init__(self, dataset, degree_node_features=False, batch_size=32, train_ratio=0.5, test_ratio=0.3):
        super(GraphClassificationDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.degree_node_features = degree_node_features
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.split_idx = None

        self.setup_node_features()

    def train_wrapper(self):
        return DataLoader(self.dataset[self.split_idx[0]], batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_wrapper(self):
        if self.split_idx[1] is not None:
            return DataLoader(self.dataset[self.split_idx[1]], batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_wrapper(self):
        return DataLoader(self.dataset[self.split_idx[2]], batch_size=self.batch_size, shuffle=False, num_workers=4)

    def setup_node_features(self):
        if self.degree_node_features or self.dataset.data[0].x is None:
            self.dataset.data = node_degree_as_feature(self.dataset.data)
        train_idx, val_idx, test_idx = split_dataset(len(self.dataset), self.train_ratio, self.test_ratio)
        self.split_idx = [train_idx, val_idx, test_idx]
