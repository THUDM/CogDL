from .. import register_data_wrapper, DataWrapper


@register_data_wrapper("multiplex_embedding_dw")
class MultiplexEmbeddingDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(MultiplexEmbeddingDataWrapper, self).__init__()

        self.dataset = dataset

    def training_wrapper(self):
        return self.dataset.data.train_data

    def test_wrapper(self):
        return self.dataset.data, self.dataset.data.test_data
