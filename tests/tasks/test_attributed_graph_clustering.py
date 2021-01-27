import torch
from cogdl.tasks import build_task
from cogdl.tasks.attributed_graph_clustering import AttributedGraphClustering
from cogdl.utils import build_args_from_dict

graph_clustering_task_name = "attributed_graph_clustering"


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "task": graph_clustering_task_name,
        "device_id": [0],
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
        "max_epoch": 3,
        "lr": 0.001,
        "T": 5,
        "gamma": 10,
    }
    return build_args_from_dict(default_dict)


def create_simple_task():
    args = get_default_args()
    args.task = graph_clustering_task_name
    args.dataset = "cora"
    args.model = "prone"
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    return AttributedGraphClustering(args)


def test_kmeans_cora():
    args = get_default_args()
    args.model_type = "content"
    args.model = "prone"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["NMI"] > 0


def test_spectral_cora():
    args = get_default_args()
    args.model_type = "content"
    args.model = "prone"
    args.dataset = "cora"
    args.cluster_method = "spectral"
    task = build_task(args)
    ret = task.train()
    assert ret["NMI"] > 0


def test_prone_cora():
    args = get_default_args()
    args.model = "prone"
    args.model_type = "spectral"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["NMI"] > 0


def test_agc_cora():
    args = get_default_args()
    args.model = "agc"
    args.model_type = "both"
    args.dataset = "cora"
    args.cluster_method = "spectral"
    args.max_iter = 2
    task = build_task(args)
    ret = task.train()
    assert ret["NMI"] > 0


def test_daegc_cora():
    args = get_default_args()
    args.model = "daegc"
    args.model_type = "both"
    args.dataset = "cora"
    args.cluster_method = "kmeans"
    task = build_task(args)
    ret = task.train()
    assert ret["NMI"] > 0


if __name__ == "__main__":
    test_kmeans_cora()
    test_spectral_cora()

    test_agc_cora()
    test_daegc_cora()

    test_prone_cora()
