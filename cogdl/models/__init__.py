import argparse
import importlib
import os

import numpy as np
import torch.nn as nn

from .base_model import BaseModel

try:
    import torch_geometric
except ImportError:
    pyg = False
    print("Failed to import PyTorch Geometric (PyG)")
else:
    pyg = True

try:
    import dgl
except ImportError:
    dgl_import = False
    print("Failed to import Deep Graph Library (DGL)")
else:
    dgl_import = True

MODEL_REGISTRY = {}


def register_model(name):
    """
    New model types can be added to cogdl with the :func:`register_model`
    function decorator.

    For example::

        @register_model('gat')
        class GAT(BaseModel):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


# automatically import any Python files in the models/ directory
for root, dirs, files in os.walk(os.path.dirname(__file__)):
    for file in files:
        if file.endswith(".py") and not file.startswith("_"):
            model_name = file[: file.find(".py")]
            if not pyg and model_name.startswith("pyg"):
                continue
            if not dgl_import and model_name.startswith("dgl"):
                continue
            model_name = os.path.join(root, model_name)
            model_name = model_name[model_name.find("models") :].replace(os.sep, ".")
            module = importlib.import_module("cogdl." + model_name)


def build_model(args):
    return MODEL_REGISTRY[args.model].build_model_from_args(args)
