import os
import importlib

from .base_model_wrapper import ModelWrapper, EmbeddingModelWrapper


MODELMODULE_REGISTRY = {}
SUPPORTED_MODELMODULES = {}


def register_model_wrapper(name):
    """
    New data wrapper types can be added to cogdl with the :func:`register_model_wrapper`
    function decorator.

    Args:
        name (str): the name of the model_wrapper
    """

    def register_model_wrapper_cls(cls):
        if name in MODELMODULE_REGISTRY:
            raise ValueError("Cannot register duplicate model_wrapper ({})".format(name))
        if not issubclass(cls, ModelWrapper):
            raise ValueError("Model ({}: {}) must extend BaseModel".format(name, cls.__name__))
        MODELMODULE_REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return register_model_wrapper_cls


def scan_model_wrappers():
    global SUPPORTED_MODELMODULES
    dirname = os.path.dirname(__file__)
    dir_names = [x for x in os.listdir(dirname) if not x.startswith("__")]
    dirs = [os.path.join(dirname, x) for x in dir_names]
    dirs_names = [(x, y) for x, y in zip(dirs, dir_names) if os.path.isdir(x)]
    dw_dict = SUPPORTED_MODELMODULES
    for _dir, _name in dirs_names:
        files = os.listdir(_dir)
        # files = [x for x in os.listdir(_dir) if os.path.isfile(x)]
        dw = [x.split(".")[0] for x in files]
        dw = [x for x in dw if not x.startswith("__")]
        path = [f"cogdl.wrappers.model_wrapper.{_name}.{x}" for x in dw]
        for x, y in zip(dw, path):
            dw_dict[x] = y


def try_import_model_wrapper(name):
    if name in MODELMODULE_REGISTRY:
        return
    if name in SUPPORTED_MODELMODULES:
        importlib.import_module(SUPPORTED_MODELMODULES[name])
    else:
        raise NotImplementedError(f"`{name}` model_wrapper is not implemented.")


def fetch_model_wrapper(name):
    try_import_model_wrapper(name)
    return MODELMODULE_REGISTRY[name]


scan_model_wrappers()
