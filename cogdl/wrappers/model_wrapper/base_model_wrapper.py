from abc import abstractmethod
import torch
from cogdl.wrappers.tools.wrapper_utils import merge_batch_indexes
from cogdl.data import Dataset


class ModelWrapper(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self):
        super(ModelWrapper, self).__init__()
        self.__model_keys__ = None
        self._loss_fn = None
        self._evaluator = None
        self.__record__ = dict()

    def forward(
        self,
    ):
        pass

    def pre_stage(self, stage, data_w):
        pass

    def post_stage(self, stage, data_w):
        pass

    def train_step(self, subgraph):
        pass

    def val_step(self, subgraph):
        pass

    def test_step(self, subgraph):
        pass

    def evaluate(self, dataset: Dataset):
        pass

    def on_train_step(self, *args, **kwargs):
        return self.train_step(*args, **kwargs)

    def on_val_step(self, *args, **kwargs):
        out = self.val_step(*args, **kwargs)
        self.set_notes(out, "val")

    def on_test_step(self, *args, **kwargs):
        out = self.test_step(*args, **kwargs)
        self.set_notes(out, "test")

    def on_evaluate(self, *args, **kwargs):
        out = self.evaluate(*args, **kwargs)
        self.set_notes(out, "evaluate")

    def set_notes(self, out, prefix="val"):
        if isinstance(out, dict):
            for key, val in out.items():
                self.note(key, val)
        elif isinstance(out, tuple) or isinstance(out, list):
            for i, val in enumerate(out):
                self.note(f"{prefix}_{i}", val)

    def note(self, name: str, data):
        if name not in self.__record__:
            name = name.lower()
            self.__record__[name] = [data]
        else:
            self.__record__[name].append(data)

    def collect_notes(self):
        if len(self.__record__) == 0:
            return None
        out = dict()
        for key, val in self.__record__.items():
            _key, _val = merge_batch_indexes(key, val)
            out[_key] = _val
        self.__record__ = dict()
        return out

    @property
    def default_loss_fn(self):
        if self._loss_fn is None:
            raise RuntimeError("`loss_fn` must be set for your ModelWrapper using `mw.default_loss_fn = your_loss_fn`.")
        return self._loss_fn

    @property
    def default_evaluator(self):
        if self._evaluator is None:
            raise RuntimeError(
                "`evaluator` must be set for your ModelWrapper using " "`mw.default_evaluator = your_evaluator`."
            )
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

    @property
    def device(self):
        # for k in self._model_key_:
        #     return next(getattr(self, k).parameters()).device
        return next(self.parameters()).device

    def load_checkpoint(self, path):
        pass

    def save_checkpoint(self, path):
        pass

    def _find_model(self):
        models = []
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                models.append(k)
        self.__model_keys__ = models

    @property
    def wrapped_model(self):
        if hasattr(self, "model"):
            return getattr(self, "model")
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
