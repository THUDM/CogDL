from .. import DataWrapper


class HeterogeneousGNNDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(HeterogeneousGNNDataWrapper, self).__init__(dataset=dataset)

        self.dataset = dataset

    def train_wrapper(self):
        return self.dataset

    def val_wrapper(self):
        return self.dataset

    def test_wrapper(self):
        return self.dataset
