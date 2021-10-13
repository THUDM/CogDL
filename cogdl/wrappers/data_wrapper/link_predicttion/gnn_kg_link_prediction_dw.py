from .. import register_data_wrapper, DataWrapper


@register_data_wrapper("gnn_kg_link_prediction_dw")
class GNNKGLinkPredictionModelWrapper(DataWrapper):
    def __init__(self, dataset):
        super(GNNKGLinkPredictionModelWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.edge_set = None

    def train_wrapper(self):
        return self.dataset.data

    def val_wrapper(self):
        return self.dataset.data

    def test_wrapper(self):
        return self.dataset.data
