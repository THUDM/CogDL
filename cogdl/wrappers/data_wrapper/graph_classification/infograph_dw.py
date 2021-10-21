from .graph_classification_dw import GraphClassificationDataWrapper


class InfoGraphDataWrapper(GraphClassificationDataWrapper):
    def test_wrapper(self):
        return self.dataset
