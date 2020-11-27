import random
import numpy as np

import torch

from utils import print_result, set_random_seed, get_dataset
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def build_default_args_for_node_classification(dataset, missing_rate=0, num_layers=40):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.005,
        "weight_decay": 5e-4,
        "max_epoch": 1000,
        "patience": 1000,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0, 1, 2, 3, 4],
        "missing_rate": missing_rate,
        "norm_mode": "PN",
        "norm_scale": 10,
        "dropout": 0.6,
        "num_layers": num_layers,
        "task": "node_classification",
        "model": "sgcpn",
        "dataset": dataset
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func
    return register_func_name


@register_func("cora")
def cora_config(args):
    return args


@register_func("citeseer")
def citeseer_config(args):
    return args


@register_func("pubmed")
def pubmed_config(args):
    return args


def run(dataset_name, missing_rate=0, num_layers=40):
    args = build_default_args_for_node_classification(dataset_name, missing_rate=missing_rate, num_layers=num_layers)
    args = DATASET_REGISTRY[dataset_name](args)
    dataset, args = get_dataset(args)
    results = []
    for seed in args.seed:
        set_random_seed(seed)
        task = build_task(args, dataset=dataset)
        result = task.train()
        results.append(result)
    return results


if __name__ == "__main__":
    datasets = ["cora", "citeseer", "pubmed"]
    params0 = [("cora", 0, 7), ("citeseer", 0, 3), ("pubmed", 0, 9)]
    params20 = [("cora", 20, 7), ("citeseer", 20, 3), ("pubmed", 20, 7)]
    params40 = [("cora", 40, 7), ("citeseer", 40, 4), ("pubmed", 40, 60)]
    params60 = [("cora", 60, 20), ("citeseer", 60, 5), ("pubmed", 60, 7)]
    params80 = [("cora", 80, 25), ("citeseer", 80, 50), ("pubmed", 80, 60)]
    params100 = [("cora", 100, 40), ("citeseer", 100, 50), ("pubmed", 100, 40)]
    results = []
    for param in params0:
        results += run(dataset_name=param[0], missing_rate=param[1], num_layers=param[2])
    print_result(results, datasets, "sgcpn")
