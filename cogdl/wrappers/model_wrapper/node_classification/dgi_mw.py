from .. import register_model_wrapper, ModelWrapper


@register_model_wrapper("dgi_mw")
class DGIModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg):
        super(DGIModelWrapper, self).__init__()
        self.model = model

    def train_step(self, batch):
        pass

    def test_step(self, batch):
        pass
