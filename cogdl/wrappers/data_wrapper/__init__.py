from .base_data_wrapper import DataWrapper
import os
import importlib


DATAMODULE_REGISTRY = {}
SUPPORTED_DATAMODULE = {}


def register_data_wrapper(name):
    """
    New data wrapper types can be added to cogdl with the :func:`register_data_wrapper`
    function decorator.

    Args:
        name (str): the name of the data_wrapper
    """

    def register_data_wrapper_cls(cls):
        if name in DATAMODULE_REGISTRY:
            raise ValueError("Cannot register duplicate data_wrapper ({})".format(name))
        if not issubclass(cls, DataWrapper):
            raise ValueError("({}: {}) must extend DataWrapper".format(name, cls.__name__))
        DATAMODULE_REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return register_data_wrapper_cls


def scan_data_wrappers():
    global SUPPORTED_DATAMODULE
    dirname = os.path.dirname(__file__)
    dir_names = [x for x in os.listdir(dirname) if not x.startswith("__")]
    dirs = [os.path.join(dirname, x) for x in dir_names]
    dirs_names = [(x, y) for x, y in zip(dirs, dir_names) if os.path.isdir(x)]
    dw_dict = SUPPORTED_DATAMODULE
    for _dir, _name in dirs_names:
        files = os.listdir(_dir)
        dw = [x.split(".")[0] for x in files]
        dw = [x for x in dw if not x.startswith("__")]
        path = [f"cogdl.wrappers.data_wrapper.{_name}.{x}" for x in dw]
        for x, y in zip(dw, path):
            dw_dict[x] = y


def try_import_data_wrapper(name):
    if name in DATAMODULE_REGISTRY:
        return
    if name in SUPPORTED_DATAMODULE:
        importlib.import_module(SUPPORTED_DATAMODULE[name])
    else:
        raise NotImplementedError(f"`{name}` data_wrapper is not implemented.")


def fetch_data_wrapper(name):
    try_import_data_wrapper(name)
    return DATAMODULE_REGISTRY[name]


scan_data_wrappers()
