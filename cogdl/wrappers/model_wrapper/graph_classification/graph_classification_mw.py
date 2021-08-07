from .. import register_model_wrapper, ModelWrapper


@register_model_wrapper("graph_classification_mw")
class GraphClassificationModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(GraphClassificationModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg

    def train_step(self, batch):
        pass

    def val_step(self, batch):
        pass

    def test_step(self, batch):
        pass

    def setup_optimizer(self):
        pass
