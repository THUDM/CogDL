import torch
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "device_id": [0],
        "num_clusters": 7,
        "cluster_method": "kmeans",
        "hidden_size": 16,
        "model_type": "emb",
        "momentum": 0,
        'enhance': None,
        "cpu": not cuda_available,
    }
    return build_args_from_dict(default_dict)

def test_prone_cora():
    args = get_default_args()
    args.task = "attributed_graph_clustering"
    args.dataset = "cora"
    args.model = "prone"
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Accuracy"] <= 1


if __name__ == "__main__":
    test_prone_cora()
