from .. import register_data_wrapper, DataWrapper


@register_data_wrapper("heterogeneous_embedding_dw")
class HeterogeneousEmbeddingDataWrapper(DataWrapper):
    def __init__(self, dataset):
        super(HeterogeneousEmbeddingDataWrapper, self).__init__()

        self.dataset = dataset

    def training_wrapper(self):
        return self.dataset.data

    def test_wrapper(self):
        return self.dataset.data
