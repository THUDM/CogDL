from typing import Union, Callable
from abc import abstractmethod
import torch
from cogdl.wrappers.tools.wrapper_utils import merge_batch_indexes
from cogdl.utils.evaluator import setup_evaluator, Accuracy, MultiLabelMicroF1


class ModelWrapper(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self):
        super(ModelWrapper, self).__init__()
        self.__model_keys__ = None
        self._loss_func = None
        self._evaluator = None
        self._evaluator_metric = None
        self.__record__ = dict()
        self.training_type = ""

    def forward(self):
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

    def evaluate(self, pred: torch.Tensor, labels: torch.Tensor, metric: Union[str, Callable] = "auto"):
        """
        method: str or callable function,
        """
        pred = pred.cpu()
        labels = labels.cpu()
        if self._evaluator is None:
            if metric == "auto":
                if len(labels.shape) > 1:
                    metric = "multilabel_microf1"
                    self._evaluator_metric = "micro_f1"
                else:
                    metric = "accuracy"
                    self._evaluator_metric = "acc"

            self._evaluator = setup_evaluator(metric)
            # self._evaluator_metric = metric
        return self._evaluator(pred, labels)

    @abstractmethod
    def setup_optimizer(self):
        raise NotImplementedError

    def set_early_stopping(self):
        """
        Return:
            1. `str`, the monitoring metric
            2. tuple(`str`, `str`), that is, (the monitoring metric, `small` or `big`). The second parameter means,
                `the smaller, the better` or `the bigger, the better`
        """
        return "val_metric", ">"

    def on_train_step(self, *args, **kwargs):
        return self.train_step(*args, **kwargs)

    def on_val_step(self, *args, **kwargs):
        out = self.val_step(*args, **kwargs)
        self.set_notes(out, "val")

    def on_test_step(self, *args, **kwargs):
        out = self.test_step(*args, **kwargs)
        self.set_notes(out, "test")

    def set_notes(self, out, prefix="val"):
        if isinstance(out, dict):
            for key, val in out.items():
                self.note(key, val)
        elif isinstance(out, tuple) or isinstance(out, list):
            for i, val in enumerate(out):
                self.note(f"{prefix}_{i}", val)

    def note(self, name: str, data, merge="mean"):
        """
        name: str
        data: Any
        """
        if name not in self.__record__:
            name = name.lower()
            self.__record__[name] = [data]
            # self.__record_merge__[name] = merge
        else:
            self.__record__[name].append(data)

    def collect_notes(self):
        if len(self.__record__) == 0:
            return None
        out = dict()
        for key, val in self.__record__.items():
            if key.endswith("_metric"):
                _val = self._evaluator.evaluate()
                if _val == 0:
                    _val = val[0]
            # elif isinstance(self._evaluator_metric, str) and key.endswith(self._evaluator_metric):
            #     _val = self._evaluator.evaluate()
            else:
                _val = merge_batch_indexes(val)
            out[key] = _val
        self.__record__ = dict()
        return out

    @property
    def default_loss_fn(self):
        if self._loss_func is None:
            raise RuntimeError(
                "`loss_fn` must be set for your ModelWrapper using `mw.default_loss_fn = your_loss_fn`.",
                f"Now self.loss_fn is {self._loss_fn}",
            )
        return self._loss_func

    @default_loss_fn.setter
    def default_loss_fn(self, loss_fn):
        self._loss_func = loss_fn

    @property
    def default_evaluator(self):
        return self._evaluator

    @default_evaluator.setter
    def default_evaluator(self, x):
        self._evaluator = x

    @property
    def device(self):
        # for k in self._model_key_:
        #     return next(getattr(self, k).parameters()).device
        return next(self.parameters()).device

    @property
    def evaluation_metric(self):
        return self._evaluator_metric

    def set_evaluation_metric(self):
        if isinstance(self._evaluator, MultiLabelMicroF1):
            self._evaluator_metric = "micro_f1"
        elif isinstance(self._evaluator, Accuracy):
            self._evaluator_metric = "acc"
        else:
            evaluation_metric = self.set_early_stopping()
            if not isinstance(evaluation_metric, str):
                evaluation_metric = evaluation_metric[0]
            if evaluation_metric.startswith("val"):
                evaluation_metric = evaluation_metric[3:]
            self._evaluator_metric = evaluation_metric

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


class UnsupervisedModelWrapper(ModelWrapper):
    def __init__(self):
        super(UnsupervisedModelWrapper, self).__init__()
        self.training_type = "unsupervised"
