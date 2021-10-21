from .. import DataWrapper
from cogdl.data import Graph


class FullBatchNodeClfDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(FullBatchNodeClfDataWrapper, self).__init__(dataset)
        self.dataset = dataset

    def train_wrapper(self) -> Graph:
        return self.dataset.data

    def val_wrapper(self):
        return self.dataset.data

    def test_wrapper(self):
        return self.dataset.data

    def pre_transform(self):
        self.dataset.data.add_remaining_self_loops()
