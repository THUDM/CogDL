import importlib
import torch
import inspect

from cogdl.data.dataset import Dataset
from .customized_data import NodeDataset, GraphDataset, generate_random_graph


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
        print("The `register_dataset` API is deprecated!")
        return cls

    return register_dataset_cls


def try_adding_dataset_args(dataset, parser):
    if dataset in SUPPORTED_DATASETS:
        path = ".".join(SUPPORTED_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = SUPPORTED_DATASETS[dataset].split(".")[-1]
        dataset_class = getattr(module, class_name)
        if hasattr(dataset_class, "add_args"):
            dataset_class.add_args(parser)


def build_dataset_from_name(dataset, split=0):
    if isinstance(dataset, list):
        dataset = dataset[0]
    if isinstance(split, list):
        split = split[0]
    if dataset in SUPPORTED_DATASETS:
        path = ".".join(SUPPORTED_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        dataset = build_dataset_from_path(dataset)
        if dataset is not None:
            return dataset
        raise NotImplementedError(f"Failed to import {dataset} dataset.")
    class_name = SUPPORTED_DATASETS[dataset].split(".")[-1]
    dataset_class = getattr(module, class_name)
    for key in inspect.signature(dataset_class.__init__).parameters.keys():
        if key == "split":
            return dataset_class(split=split)

    return dataset_class()


def build_dataset_pretrain(args):
    args.pretrain = False
    dataset_names = args.dataset
    if ' ' in args.dataset:
        datasets_name = args.dataset.split(' ')
        dataset = []
        for dataset_ in datasets_name:
            args.dataset = dataset_
            dataset.append(build_dataset(args))
    else:
        dataset = [build_dataset(args)]
    args.pretrain = True
    args.dataset = dataset_names
    dataset_class = getattr(importlib.import_module("cogdl.datasets.gcc_data"), "PretrainDataset")
    return dataset_class(args.dataset, [x.data for x in dataset])


def build_dataset(args):
    if not hasattr(args, "split"):
        args.split = 0
    if not hasattr(args, "pretrain") or not args.pretrain:
        dataset = build_dataset_from_name(args.dataset, args.split)
    else:
        dataset = build_dataset_pretrain(args)

    if hasattr(dataset, "num_classes") and dataset.num_classes > 0:
        args.num_classes = dataset.num_classes
    if hasattr(dataset, "num_features") and dataset.num_features > 0:
        args.num_features = dataset.num_features

    return dataset


def build_dataset_from_path(data_path, dataset=None):
    if dataset is not None and dataset in SUPPORTED_DATASETS:
        path = ".".join(SUPPORTED_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = SUPPORTED_DATASETS[dataset].split(".")[-1]
        dataset_class = getattr(module, class_name)
        keys = inspect.signature(dataset_class.__init__).parameters.keys()
        if "data_path" in keys:
            dataset = dataset_class(data_path=data_path)
        elif "root" in keys:
            dataset = dataset_class(root=data_path)
        return dataset

    if dataset is None:
        try:
            return torch.load(data_path)
        except Exception as e:
            print(e)
            exit(0)
    raise ValueError("You are expected to specify `dataset` and `data_path`")


SUPPORTED_DATASETS = {
    "gcc_academic": "cogdl.datasets.gcc_data.Academic_GCCDataset",
    "gcc_dblp_netrep": "cogdl.datasets.gcc_data.DBLPNetrep_GCCDataset",
    "gcc_dblp_snap": "cogdl.datasets.gcc_data.DBLPSnap_GCCDataset",
    "gcc_facebook": "cogdl.datasets.gcc_data.Facebook_GCCDataset",
    "gcc_imdb": "cogdl.datasets.gcc_data.IMDB_GCCDataset",
    "gcc_livejournal": "cogdl.datasets.gcc_data.Livejournal_GCCDataset",
    "kdd_icdm": "cogdl.datasets.gcc_data.KDD_ICDM_GCCDataset",
    "sigir_cikm": "cogdl.datasets.gcc_data.SIGIR_CIKM_GCCDataset",
    "sigmod_icde": "cogdl.datasets.gcc_data.SIGMOD_ICDE_GCCDataset",
    "usa-airport": "cogdl.datasets.gcc_data.USAAirportDataset",
    "h-index": "cogdl.datasets.gcc_data.HIndexDataset",
    "ogbn-arxiv": "cogdl.datasets.ogb.OGBArxivDataset",
    "ogbn-products": "cogdl.datasets.ogb.OGBProductsDataset",
    "ogbn-proteins": "cogdl.datasets.ogb.OGBProteinsDataset",
    "ogbn-papers100M": "cogdl.datasets.ogb.OGBPapers100MDataset",
    "ogbg-molbace": "cogdl.datasets.ogb.OGBMolbaceDataset",
    "ogbg-molhiv": "cogdl.datasets.ogb.OGBMolhivDataset",
    "ogbg-molpcba": "cogdl.datasets.ogb.OGBMolpcbaDataset",
    "ogbg-ppa": "cogdl.datasets.ogb.OGBPpaDataset",
    "ogbg-code": "cogdl.datasets.ogb.OGBCodeDataset",
    "ogbl-ppa": "cogdl.datasets.ogb.OGBLPpaDataset",
    "ogbl-ddi": "cogdl.datasets.ogb.OGBLDdiDataset",
    "ogbl-collab": "cogdl.datasets.ogb.OGBLCollabDataset",
    "ogbl-citation2": "cogdl.datasets.ogb.OGBLCitation2Dataset",
    "amazon": "cogdl.datasets.gatne.AmazonDataset",
    "twitter": "cogdl.datasets.gatne.TwitterDataset",
    "youtube": "cogdl.datasets.gatne.YouTubeDataset",
    "gtn-acm": "cogdl.datasets.gtn_data.ACM_GTNDataset",
    "gtn-dblp": "cogdl.datasets.gtn_data.DBLP_GTNDataset",
    "gtn-imdb": "cogdl.datasets.gtn_data.IMDB_GTNDataset",
    "fb13": "cogdl.datasets.kg_data.FB13Datset",
    "fb15k": "cogdl.datasets.kg_data.FB15kDatset",
    "fb15k237": "cogdl.datasets.kg_data.FB15k237Datset",
    "wn18": "cogdl.datasets.kg_data.WN18Datset",
    "wn18rr": "cogdl.datasets.kg_data.WN18RRDataset",
    "fb13s": "cogdl.datasets.kg_data.FB13SDatset",
    "cora": "cogdl.datasets.planetoid_data.CoraDataset",
    "citeseer": "cogdl.datasets.planetoid_data.CiteSeerDataset",
    "pubmed": "cogdl.datasets.planetoid_data.PubMedDataset",
    "chameleon": "cogdl.datasets.geom_data.ChameleonDataset",
    "cornell": "cogdl.datasets.geom_data.CornellDataset",
    "film": "cogdl.datasets.geom_data.FilmDataset",
    "squirrel": "cogdl.datasets.geom_data.SquirrelDataset",
    "texas": "cogdl.datasets.geom_data.TexasDataset",
    "wisconsin": "cogdl.datasets.geom_data.WisconsinDataset",
    "cora_geom": "cogdl.datasets.geom_data.CoraGeomDataset",
    "citeseer_geom": "cogdl.datasets.geom_data.CiteSeerGeomDataset",
    "pubmed_geom": "cogdl.datasets.geom_data.PubMedGeomDataset",
    "blogcatalog": "cogdl.datasets.matlab_matrix.BlogcatalogDataset",
    "flickr-ne": "cogdl.datasets.matlab_matrix.FlickrDataset",
    "dblp-ne": "cogdl.datasets.matlab_matrix.DblpNEDataset",
    "youtube-ne": "cogdl.datasets.matlab_matrix.YoutubeNEDataset",
    "wikipedia": "cogdl.datasets.matlab_matrix.WikipediaDataset",
    "ppi-ne": "cogdl.datasets.matlab_matrix.PPIDataset",
    "han-acm": "cogdl.datasets.han_data.ACM_HANDataset",
    "han-dblp": "cogdl.datasets.han_data.DBLP_HANDataset",
    "han-imdb": "cogdl.datasets.han_data.IMDB_HANDataset",
    "mutag": "cogdl.datasets.tu_data.MUTAGDataset",
    "imdb-b": "cogdl.datasets.tu_data.ImdbBinaryDataset",
    "imdb-m": "cogdl.datasets.tu_data.ImdbMultiDataset",
    "collab": "cogdl.datasets.tu_data.CollabDataset",
    "proteins": "cogdl.datasets.tu_data.ProteinsDataset",
    "reddit-b": "cogdl.datasets.tu_data.RedditBinary",
    "reddit-multi-5k": "cogdl.datasets.tu_data.RedditMulti5K",
    "reddit-multi-12k": "cogdl.datasets.tu_data.RedditMulti12K",
    "ptc-mr": "cogdl.datasets.tu_data.PTCMRDataset",
    "nci1": "cogdl.datasets.tu_data.NCI1Dataset",
    "nci109": "cogdl.datasets.tu_data.NCI109Dataset",
    "enzymes": "cogdl.datasets.tu_data.ENZYMES",
    "yelp": "cogdl.datasets.saint_data.YelpDataset",
    "amazon-s": "cogdl.datasets.saint_data.AmazonDataset",
    "flickr": "cogdl.datasets.saint_data.FlickrDataset",
    "reddit": "cogdl.datasets.saint_data.RedditDataset",
    "ppi": "cogdl.datasets.saint_data.PPIDataset",
    "ppi-large": "cogdl.datasets.saint_data.PPILargeDataset",
    "l0fos": "cogdl.datasets.oagbert_data.l0fos",
    "aff30": "cogdl.datasets.oagbert_data.aff30",
    "arxivvenue": "cogdl.datasets.oagbert_data.arxivvenue",
    "yelp2018": "cogdl.datasets.rec_data.Yelp2018Dataset",
    "ali": "cogdl.datasets.rec_data.AliDataset",
    "amazon-rec": "cogdl.datasets.rec_data.AmazonRecDataset",
    "Github": "cogdl.datasets.rd2cd_data.Github",
    "Elliptic": "cogdl.datasets.rd2cd_data.Elliptic",
    "Film": "cogdl.datasets.rd2cd_data.Film",
    "Wiki": "cogdl.datasets.rd2cd_data.Wiki",
    "Clothing": "cogdl.datasets.rd2cd_data.Clothing",
    "Electronics": "cogdl.datasets.rd2cd_data.Electronics",
    "Dblp": "cogdl.datasets.rd2cd_data.Dblp",
    "Yelpchi": "cogdl.datasets.rd2cd_data.Yelpchi",
    "Alpha": "cogdl.datasets.rd2cd_data.Alpha",
    "Weibo": "cogdl.datasets.rd2cd_data.Weibo",
    "bgp": "cogdl.datasets.rd2cd_data.bgp",
    "ssn5": "cogdl.datasets.rd2cd_data.ssn5",
    "ssn7": "cogdl.datasets.rd2cd_data.ssn7",
    "Aids": "cogdl.datasets.rd2cd_data.Aids",
    "Nba": "cogdl.datasets.rd2cd_data.Nba",
    "Pokec_z": "cogdl.datasets.rd2cd_data.Pokec_z",
    "grb-cora": "cogdl.datasets.grb_data.Cora_GRBDataset",
    "grb-citeseer": "cogdl.datasets.grb_data.Citeseer_GRBDataset",
    "grb-reddit": "cogdl.datasets.grb_data.Reddit_GRBDataset",
    "grb-aminer": "cogdl.datasets.grb_data.Aminer_GRBDataset",
    "grb-flickr": "cogdl.datasets.grb_data.Flickr_GRBDataset",
    "pems-stgcn": "cogdl.datasets.stgcn_data.PeMS_Dataset",
    "pems-stgat": "cogdl.datasets.stgat_data.PeMS_Dataset",
}
