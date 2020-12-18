import torch
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "hidden_size": 16,
        "dropout": 0.5,
        "patience": 100,
        "max_epoch": 500,
        "cpu": not cuda_available,
        "lr": 0.01,
        "device_id": [0],
        "weight_decay": 5e-4,
        "alpha": 0.05,
        "t": 5.0,
        "k": 128,
        "eps": 0.01,
        "gdc_type": "ppr",
    }
    return build_args_from_dict(default_dict)


if __name__ == "__main__":
    args = get_default_args()
    args.task = "node_classification"
    args.dataset = "cora"
    args.model = "gdc_gcn"
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 1
    args.alpha = 0.05  # ppr filter param
    # args.t = 5.0 # heat filter param
    args.k = 128  # top k entries to be retained
    args.eps = 0.01  # change depending on gdc_type
    args.dataset = dataset
    args.gdc_type = "none"  # ppr, heat, none

    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
