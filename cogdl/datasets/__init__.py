import importlib
import os

from cogdl.data.dataset import Dataset

try:
    import torch_geometric
except ImportError:
    pyg = False
else:
    pyg = True

try:
    import dgl
    from dgl.data.tu import TUDataset
except ImportError:
    dgl_import = False
else:
    dgl_import = True

DATASET_REGISTRY = {}


def register_dataset(name):
    """
    New dataset types can be added to cogdl with the :func:`register_dataset`
    function decorator.

    For example::

        @register_dataset('my_dataset')
        class MyDataset():
            (...)

    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, Dataset) and (
            pyg and not issubclass(cls, torch_geometric.data.Dataset)
        ) and (
            dgl_import and not issubclass(cls, TUDataset)
        ):
            raise ValueError(
                "Dataset ({}: {}) must extend cogdl.data.Dataset".format(
                    name, cls.__name__
                )
            )
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


# automatically import any Python files in the datasets/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        dataset_name = file[: file.find(".py")]
        if not pyg and dataset_name.startswith("pyg"):
            continue
        if not dgl_import and dataset_name.startswith("dgl"):
            continue
        module = importlib.import_module("cogdl.datasets." + dataset_name)


def build_dataset(args):
    return DATASET_REGISTRY[args.dataset]()


def build_dataset_from_name(dataset):
    return DATASET_REGISTRY[dataset]()
