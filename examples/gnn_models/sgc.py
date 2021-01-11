import torch
from cogdl import experiment
from cogdl.utils import build_args_from_dict


def default_parameter():
    default_dict = {
        "hidden_size": 16,
        "dropout": 0.5,
        "patience": 100,
        "max_epoch": 500,
        "lr": 0.01,
        "device_id": [0],
        "weight_decay": 5e-4,
        "missing_rate": -1,
    }
    return build_args_from_dict(default_dict)


def run(dataset_name):
    args = default_parameter().__dict__
    results = experiment(task="node_classification", dataset=dataset_name, model="srgcn", **args)
    return results


if __name__ == "__main__":
    run("cora")
