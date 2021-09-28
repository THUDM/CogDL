import torch

from .. import register_data_wrapper, DataWrapper
from cogdl.wrappers.tools.wrapper_utils import node_degree_as_feature, split_dataset
from cogdl.data import DataLoader
from cogdl.models.nn.patchy_san import get_single_feature


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

    def train_wrapper(self):
        return DataLoader(self.dataset[self.split_idx[0]], batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_wrapper(self):
        if self.split_idx[1] is not None:
            return DataLoader(self.dataset[self.split_idx[1]], batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_wrapper(self):
        return DataLoader(self.dataset[self.split_idx[2]], batch_size=self.batch_size, shuffle=False, num_workers=4)

    def pre_transform(self):
        if self.degree_node_features and self.dataset.data.x is None:
            self.dataset.data = node_degree_as_feature(self.dataset.data)
        train_idx, val_idx, test_idx = split_dataset(len(self.dataset), self.train_ratio, self.test_ratio)
        self.split_idx = [train_idx, val_idx, test_idx]


@register_data_wrapper("patchy_san_dw")
class PATCHY_SAN_DataWrapper(GraphClassificationDataWrapper):
    @staticmethod
    def add_args(parser):
        GraphClassificationDataWrapper.add_args(parser)
        parser.add_argument("--num-sample", default=30, type=int, help="Number of chosen vertexes")
        parser.add_argument("--num-neighbor", default=10, type=int, help="Number of neighbor in constructing features")

    def __init__(self, dataset, num_sample, num_neighbor, *args, **kwargs):
        super(PATCHY_SAN_DataWrapper, self).__init__(dataset, *args, **kwargs)
        self.sample = num_sample
        self.dataset = dataset
        self.neighbor = num_neighbor

    def pre_transform(self):
        dataset = self.dataset
        num_features = dataset.data.num_features
        num_classes = dataset.data.num_classes
        for i, data in enumerate(dataset):
            new_feature = get_single_feature(
                dataset[i], num_features, num_classes, self.sample, self.neighbor, self.stride
            )
            dataset[i].x = torch.from_numpy(new_feature)
        self.dataset = dataset
        super(PATCHY_SAN_DataWrapper, self).pre_transform()
