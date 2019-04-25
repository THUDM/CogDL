import argparse
import importlib
import os

import torch.nn as nn

from .cg_model import BaseCGModel

MODEL_REGISTRY = {}


def build_model(args):
    return MODEL_REGISTRY[args.model].build_model_from_args(args)


def register_model(name):
    """
    New model types can be added to cognitive_graph with the :func:`register_model`
    function decorator.

    For example::

        @register_model('gat')
        class GAT(BaseCGModel):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, BaseCGModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseCGModel".format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        model_name = file[: file.find(".py")]
        module = importlib.import_module("cognitive_graph.models." + model_name)
