from .. import register_data_wrapper
from .graph_classification_dw import GraphClassificationDataWrapper


@register_data_wrapper("infograph_dw")
class InfoGraphDataWrapper(GraphClassificationDataWrapper):
    def test_wrapper(self):
        return self.dataset
