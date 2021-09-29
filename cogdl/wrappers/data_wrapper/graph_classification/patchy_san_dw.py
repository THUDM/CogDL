import torch

from .. import register_data_wrapper
from .graph_classification_dw import GraphClassificationDataWrapper
from cogdl.models.nn.patchy_san import get_single_feature


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
