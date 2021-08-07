from abc import abstractmethod
import torch


class ModelWrapper(object):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self):
        self.__model_keys__ = None
        self._loss_fn = None
        self._evaluator = None

    def train_step(self, batch):
        pass

    def val_step(self, batch):
        pass

    def test_step(self, batch):
        pass

    @property
    def default_loss_fn(self):
        if self._loss_fn is None:
            raise RuntimeError("`loss_fn` must be set for your ModelWrapper using `mw.loss_fn = your_loss_fn`.")
        return self._loss_fn

    @property
    def default_evaluator(self):
        if self._evaluator is None:
            raise RuntimeError("`evaluator` must be set for your ModelWrapper using `mw.evaluator = your_evaluator`.")
        return self._evaluator

    @abstractmethod
    def setup_optimizer(self):
        raise NotImplementedError

    @default_loss_fn.setter
    def default_loss_fn(self, loss_fn):
        self._loss_fn = loss_fn

    @default_evaluator.setter
    def default_evaluator(self, evaluator):
        self._evaluator = evaluator

    def checkpoint(self):
        pass

    def to(self, device):
        for k in self._model_key_:
            getattr(self, k).to(device)

    def cuda(self):
        for k in self._model_key_:
            getattr(self, k).cuda()

    def cpu(self):
        for k in self._model_key_:
            getattr(self, k).cpu()

    def train(self):
        for k in self._model_key_:
            getattr(self, k).train()

    def eval(self):
        for k in self._model_key_:
            getattr(self, k).eval()

    def parameters(self):
        if len(self._model_key_) == 1:
            return getattr(self, self._model_key_[0]).parameters()
        else:
            params = ([{"params": getattr(self, k).parameters()} for k in self._model_key_],)
            return params

    def _find_model(self):
        models = []
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                models.append(k)
        self.__model_keys__ = models

    @property
    def wrapped_model(self):
        assert len(self._model_key_) == 1, f"{len(self._model_key_)} exists"
        return getattr(self, self._model_key_[0])

    @wrapped_model.setter
    def wrapped_model(self, model):
        if len(self._model_key_) == 0:
            self.__model_keys__ = [None]
        setattr(self, self._model_key_[0], model)

    @property
    def _model_key_(self):
        if self.__model_keys__ is None:
            self._find_model()
        return self.__model_keys__


class EmbeddingModelWrapper(ModelWrapper):
    def setup_optimizer(self):
        pass
