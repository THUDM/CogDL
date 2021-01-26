import torch
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "hidden_size": 16,
        "dropout": 0.5,
        "patience": 2,
        "max_epoch": 5,
        "batch_size": 20,
        "cpu": not cuda_available,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "checkpoint": False,
        "device_id": [0],
    }
    return build_args_from_dict(default_dict)


def test_fastgcn_cora():
    args = get_default_args()
    args.task = "node_classification_sampling"
    args.dataset = "cora"
    args.model = "fastgcn"
    args.num_layers = 3
    args.sample_size = [512, 256, 256]
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] >= 0 and ret["Acc"] <= 1


def test_asgcn_cora():
    args = get_default_args()
    args.task = "node_classification_sampling"
    args.dataset = "cora"
    args.model = "asgcn"
    args.num_layers = 3
    args.sample_size = [64, 64, 32]
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] >= 0 and ret["Acc"] <= 1


if __name__ == "__main__":
    test_fastgcn_cora()
    test_asgcn_cora()
