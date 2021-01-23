import importlib

from cogdl.data.dataset import Dataset

try:
    import torch_geometric
except ImportError:
    pyg = False
else:
    pyg = True


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
        # if name in DATASET_REGISTRY:
        #     raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, Dataset) and (pyg and not issubclass(cls, torch_geometric.data.Dataset)):
            raise ValueError("Dataset ({}: {}) must extend cogdl.data.Dataset".format(name, cls.__name__))
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def try_import_dataset(dataset):
    if dataset not in DATASET_REGISTRY:
        if dataset in SUPPORTED_DATASETS:
            importlib.import_module(SUPPORTED_DATASETS[dataset])
        else:
            print(f"Failed to import {dataset} dataset.")
            return False
    return True


def build_dataset(args):
    if not try_import_dataset(args.dataset):
        exit(1)
    return DATASET_REGISTRY[args.dataset](args=args)


def build_dataset_from_name(dataset):
    if not try_import_dataset(dataset):
        exit(1)
    return DATASET_REGISTRY[dataset]()


SUPPORTED_DATASETS = {
    "kdd_icdm": "cogdl.datasets.gcc_data",
    "sigir_cikm": "cogdl.datasets.gcc_data",
    "sigmod_icde": "cogdl.datasets.gcc_data",
    "usa-airport": "cogdl.datasets.gcc_data",
    "test_small": "cogdl.datasets.test_data",
    "ogbn-arxiv": "cogdl.datasets.pyg_ogb",
    "ogbn-products": "cogdl.datasets.pyg_ogb",
    "ogbn-proteins": "cogdl.datasets.pyg_ogb",
    "ogbn-mag": "cogdl.datasets.pyg_ogb",
    "ogbn-papers100M": "cogdl.datasets.pyg_ogb",
    "ogbg-molbace": "cogdl.datasets.pyg_ogb",
    "ogbg-molhiv": "cogdl.datasets.pyg_ogb",
    "ogbg-molpcba": "cogdl.datasets.pyg_ogb",
    "ogbg-ppa": "cogdl.datasets.pyg_ogb",
    "ogbg-code": "cogdl.datasets.pyg_ogb",
    "amazon": "cogdl.datasets.gatne",
    "twitter": "cogdl.datasets.gatne",
    "youtube": "cogdl.datasets.gatne",
    "gtn-acm": "cogdl.datasets.gtn_data",
    "gtn-dblp": "cogdl.datasets.gtn_data",
    "gtn-imdb": "cogdl.datasets.gtn_data",
    "fb13": "cogdl.datasets.kg_data",
    "fb15k": "cogdl.datasets.kg_data",
    "fb15k237": "cogdl.datasets.kg_data",
    "wn18": "cogdl.datasets.kg_data",
    "wn18rr": "cogdl.datasets.kg_data",
    "fb13s": "cogdl.datasets.kg_data",
    "cora": "cogdl.datasets.planetoid_data",
    "citeseer": "cogdl.datasets.planetoid_data",
    "pubmed": "cogdl.datasets.planetoid_data",
    "blogcatalog": "cogdl.datasets.matlab_matrix",
    "flickr-ne": "cogdl.datasets.matlab_matrix",
    "wikipedia": "cogdl.datasets.matlab_matrix",
    "ppi": "cogdl.datasets.matlab_matrix",
    "han-acm": "cogdl.datasets.han_data",
    "han-dblp": "cogdl.datasets.han_data",
    "han-imdb": "cogdl.datasets.han_data",
    "mutag": "cogdl.datasets.tu_data",
    "imdb-b": "cogdl.datasets.tu_data",
    "imdb-m": "cogdl.datasets.tu_data",
    "collab": "cogdl.datasets.tu_data",
    "proteins": "cogdl.datasets.tu_data",
    "reddit-b": "cogdl.datasets.tu_data",
    "reddit-multi-5k": "cogdl.datasets.tu_data",
    "reddit-multi-12k": "cogdl.datasets.tu_data",
    "ptc-mr": "cogdl.datasets.tu_data",
    "nci1": "cogdl.datasets.tu_data",
    "nci109": "cogdl.datasets.tu_data",
    "enzymes": "cogdl.datasets.tu_data",
    "yelp": "cogdl.datasets.saint_data",
    "amazon-s": "cogdl.datasets.saint_data",
    "flickr": "cogdl.datasets.saint_data",
    "reddit": "cogdl.datasets.saint_data",
    "test_bio": "cogdl.datasets.pyg_strategies_data",
    "test_chem": "cogdl.datasets.pyg_strategies_data",
    "bio": "cogdl.datasets.pyg_strategies_data",
    "chem": "cogdl.datasets.pyg_strategies_data",
    "bace": "cogdl.datasets.pyg_strategies_data",
    "bbbp": "cogdl.datasets.pyg_strategies_data",
}
