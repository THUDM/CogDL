import importlib

from cogdl.data.dataset import Dataset
from .customized_data import NodeDataset, GraphDataset

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
        assert hasattr(args, "task")
        dataset = build_dataset_from_path(args.dataset, args.task)
        if dataset is not None:
            return dataset
        exit(1)
    else:
        dataset = DATASET_REGISTRY[args.dataset]()
    if dataset.num_classes > 0:
        args.num_classes = dataset.num_classes
    if dataset.num_features > 0:
        args.num_features = dataset.num_features
    return dataset


def build_dataset_from_name(dataset):
    if not try_import_dataset(dataset):
        exit(1)
    return DATASET_REGISTRY[dataset]()


def build_dataset_from_path(data_path, task=None, dataset=None):
    if dataset is not None and dataset in SUPPORTED_DATASETS:
        if try_import_dataset(dataset):
            return DATASET_REGISTRY[dataset](data_path=data_path)

    if task is None:
        return None
    if "node_classification" in task:
        return NodeDataset(data_path)
    elif "graph_classification" in task:
        return GraphDataset(data_path)
    else:
        return None


SUPPORTED_DATASETS = {
    "kdd_icdm": "cogdl.datasets.gcc_data",
    "sigir_cikm": "cogdl.datasets.gcc_data",
    "sigmod_icde": "cogdl.datasets.gcc_data",
    "usa-airport": "cogdl.datasets.gcc_data",
    "test_small": "cogdl.datasets.test_data",
    "ogbn-arxiv": "cogdl.datasets.ogb",
    "ogbn-products": "cogdl.datasets.ogb",
    "ogbn-proteins": "cogdl.datasets.ogb",
    "ogbn-mag": "cogdl.datasets.ogb",
    "ogbn-papers100M": "cogdl.datasets.ogb",
    "ogbg-molbace": "cogdl.datasets.ogb",
    "ogbg-molhiv": "cogdl.datasets.ogb",
    "ogbg-molpcba": "cogdl.datasets.ogb",
    "ogbg-ppa": "cogdl.datasets.ogb",
    "ogbg-code": "cogdl.datasets.ogb",
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
    "dblp-ne": "cogdl.datasets.matlab_matrix",
    "youtube-ne": "cogdl.datasets.matlab_matrix",
    "wikipedia": "cogdl.datasets.matlab_matrix",
    "ppi-ne": "cogdl.datasets.matlab_matrix",
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
    "ppi": "cogdl.datasets.saint_data",
    "ppi-large": "cogdl.datasets.saint_data",
    "test_bio": "cogdl.datasets.strategies_data",
    "test_chem": "cogdl.datasets.strategies_data",
    "bio": "cogdl.datasets.strategies_data",
    "chem": "cogdl.datasets.strategies_data",
    "bace": "cogdl.datasets.strategies_data",
    "bbbp": "cogdl.datasets.strategies_data",
    "l0fos": "cogdl.datasets.oagbert_data",
    "aff30": "cogdl.datasets.oagbert_data",
    "arxivvenue": "cogdl.datasets.oagbert_data",
    "yelp2018": "cogdl.datasets.rec_data",
    "ali": "cogdl.datasets.rec_data",
    "amazon-rec": "cogdl.datasets.rec_data",
}
