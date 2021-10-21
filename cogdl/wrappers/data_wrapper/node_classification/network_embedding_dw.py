import numpy as np

from .. import DataWrapper


class NetworkEmbeddingDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(NetworkEmbeddingDataWrapper, self).__init__()

        self.dataset = dataset
        data = dataset[0]

        num_nodes = data.num_nodes
        num_classes = dataset.num_classes
        if len(data.y.shape) > 1:
            self.label_matrix = data.y
        else:
            self.label_matrix = np.zeros((num_nodes, num_classes), dtype=int)
            self.label_matrix[range(num_nodes), data.y.numpy()] = 1

    def train_wrapper(self):
        return self.dataset.data

    def test_wrapper(self):
        return self.label_matrix
