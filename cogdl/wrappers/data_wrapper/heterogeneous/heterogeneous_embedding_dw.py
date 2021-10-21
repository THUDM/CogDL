from .. import DataWrapper


class HeterogeneousEmbeddingDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(HeterogeneousEmbeddingDataWrapper, self).__init__()

        self.dataset = dataset

    def train_wrapper(self):
        return self.dataset.data

    def test_wrapper(self):
        return self.dataset.data
