import importlib
import os

from cogdl.data.dataset import Dataset

DATASET_REGISTRY = {}


def build_dataset(args):
    return DATASET_REGISTRY[args.dataset]()


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
        if not issubclass(cls, Dataset):
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
        module = importlib.import_module("cogdl.datasets." + dataset_name)
