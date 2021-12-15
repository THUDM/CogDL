import torch
from cogdl.options import get_default_args
from cogdl.experiments import train

cuda_available = torch.cuda.is_available()
default_dict = {
    "devices": [0],
    "num_clusters": 7,
    "cluster_method": "kmeans",
    "evaluate": "NMI",
    "hidden_size": 16,
    "model_type": "spectral",
    "enhance": None,
    "cpu": not cuda_available,
    "step": 5,
    "theta": 0.5,
    "mu": 0.2,
    "checkpoint": False,
    "walk_length": 10,
    "walk_num": 4,
    "window_size": 5,
    "worker": 2,
    "iteration": 3,
    "rank": 64,
    "negative": 1,
    "is_large": False,
    "max_iter": 5,
    "embedding_size": 16,
    "weight_decay": 0.01,
    "num_heads": 1,
    "dropout": 0,
    "epochs": 3,
    "lr": 0.001,
    "T": 5,
    "gamma": 10,
    "n_warmup_steps": 0,
}


def get_default_args_agc(dataset, model, dw=None, mw=None):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args


def test_kmeans_cora():
    args = get_default_args_agc(dataset="cora", model="prone", mw="agc_mw", dw="node_classification_dw")
    args.model_type = "content"
    args.cluster_method = "kmeans"
    ret = train(args)
    assert ret["nmi"] >= 0


def test_spectral_cora():
    args = get_default_args_agc(dataset="cora", model="prone", mw="agc_mw", dw="node_classification_dw")
    args.model_type = "content"
    args.cluster_method = "spectral"
    ret = train(args)
    assert ret["nmi"] >= 0


def test_prone_cora():
    args = get_default_args_agc(dataset="cora", model="prone", mw="agc_mw", dw="node_classification_dw")
    args.model_type = "spectral"
    args.cluster_method = "kmeans"
    ret = train(args)
    assert ret["nmi"] >= 0


def test_agc_cora():
    args = get_default_args_agc(dataset="cora", model="agc", mw="agc_mw", dw="node_classification_dw")
    args.model_type = "both"
    args.cluster_method = "spectral"
    args.max_iter = 1
    ret = train(args)
    assert ret["nmi"] >= 0


def test_daegc_cora():
    args = get_default_args_agc(dataset="cora", model="daegc", mw="daegc_mw", dw="node_classification_dw")
    args.model_type = "both"
    args.cluster_method = "kmeans"
    ret = train(args)
    assert ret["nmi"] >= 0


def test_gae_cora():
    args = get_default_args_agc(dataset="cora", model="gae", mw="gae_mw", dw="node_classification_dw")
    args.num_layers = 2
    args.model_type = "both"
    args.cluster_method = "kmeans"
    ret = train(args)
    assert ret["nmi"] >= 0


def test_vgae_cora():
    args = get_default_args_agc(dataset="cora", model="vgae", mw="gae_mw", dw="node_classification_dw")
    args.model_type = "both"
    args.cluster_method = "kmeans"
    args.epochs = 1
    ret = train(args)
    assert ret["nmi"] >= 0


if __name__ == "__main__":
    test_gae_cora()
    test_vgae_cora()

    test_kmeans_cora()
    test_spectral_cora()

    test_agc_cora()
    test_daegc_cora()

    test_prone_cora()
