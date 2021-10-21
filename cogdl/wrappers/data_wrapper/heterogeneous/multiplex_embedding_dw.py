from .. import DataWrapper


class MultiplexEmbeddingDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(MultiplexEmbeddingDataWrapper, self).__init__()

        self.dataset = dataset

    def train_wrapper(self):
        return self.dataset.data.train_data

    def test_wrapper(self):
        return self.dataset.data.test_data
