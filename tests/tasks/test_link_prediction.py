import torch

from cogdl.options import get_default_args
from cogdl.experiments import train


default_dict_emb_link = {
    "hidden_size": 16,
    "negative_ratio": 3,
    "patience": 1,
    "epochs": 1,
    "cpu": True,
    "checkpoint": False,
    "save_dir": ".",
    "device_id": [0],
    "activation": "relu",
    "residual": False,
    "norm": None,
    "actnn": False,
}


def get_default_args_emb_link(dataset, model, dw=None, mw=None):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict_emb_link.items():
        args.__setattr__(key, value)
    return args


def test_prone_ppi():
    args = get_default_args_emb_link("ppi-ne", "prone", "embedding_link_prediction_dw", "embedding_link_prediction_mw")
    args.step = 3
    args.theta = 0.5
    args.mu = 0.2
    ret = train(args)
    assert 0 <= ret["ROC_AUC"] <= 1


default_dict_kg = {
    "epochs": 1,
    "num_bases": 4,
    "num_layers": 2,
    "hidden_size": 16,
    "penalty": 0.001,
    "sampling_rate": 0.001,
    "dropout": 0.3,
    "evaluate_interval": 2,
    "patience": 20,
    "lr": 0.001,
    "weight_decay": 0,
    "negative_ratio": 3,
    "cpu": True,
    "checkpoint": False,
    "save_dir": ".",
    "device_id": [0],
    "activation": "relu",
    "residual": False,
    "norm": None,
    "actnn": False,
}


def get_default_args_kg(dataset, model, dw="gnn_kg_link_prediction_dw", mw="gnn_kg_link_prediction_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict_kg.items():
        args.__setattr__(key, value)
    return args


def test_rgcn_wn18():
    args = get_default_args_kg(dataset="wn18", model="rgcn")
    args.self_dropout = 0.2
    args.self_loop = True
    args.regularizer = "basis"
    ret = train(args)
    assert 0 <= ret["mrr"] <= 1


def test_compgcn_wn18rr():
    args = get_default_args_kg(dataset="wn18rr", model="compgcn")
    args.lbl_smooth = 0.1
    args.score_func = "distmult"
    args.regularizer = "basis"
    args.opn = "sub"
    ret = train(args)
    assert 0 <= ret["mrr"] <= 1


default_dict_gnn_link = {
    "hidden_size": 32,
    "dataset": "cora",
    "model": "gcn",
    "task": "link_prediction",
    "lr": 0.005,
    "weight_decay": 5e-4,
    "epochs": 60,
    "patience": 2,
    "num_layers": 2,
    "evaluate_interval": 1,
    "cpu": True,
    "device_id": [0],
    "dropout": 0.5,
    "checkpoint": False,
    "save_dir": ".",
    "activation": "relu",
    "residual": False,
    "norm": None,
    "actnn": False,
}


def get_default_args_gnn_link(dataset, model, dw="gnn_link_prediction_dw", mw="gnn_link_prediction_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict_gnn_link.items():
        args.__setattr__(key, value)
    return args


def test_gcn_cora():
    args = get_default_args_gnn_link("cora", "gcn")
    ret = train(args)
    assert 0.5 <= ret["auc"] <= 1.0


if __name__ == "__main__":
    test_prone_ppi()

    test_rgcn_wn18()
    test_compgcn_wn18rr()

    test_gcn_cora()
