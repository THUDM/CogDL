from .. import DataWrapper


class GNNKGLinkPredictionDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(GNNKGLinkPredictionDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.edge_set = None

    def train_wrapper(self):
        return self.dataset.data

    def val_wrapper(self):
        return self.dataset.data

    def test_wrapper(self):
        return self.dataset.data
