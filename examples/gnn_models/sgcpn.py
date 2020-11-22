import random
import numpy as np

import torch

from utils import print_result, set_random_seed, get_dataset
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}


def build_default_args_for_node_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.005,
        "weight_decay": 5e-4,
        "max_epoch": 1000,
        "patience": 1000,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0, 1, 2, 3, 4],
        "norm_mode": "PN",
        "norm_scale": 10,
        "dropout": 0.6,
        "num_layers": 40,
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


@register_func("cora-missing-0")
def cora_config(args):
    return args


@register_func("cora-missing-20")
def cora_config(args):
    return args


@register_func("cora-missing-40")
def cora_config(args):
    return args


@register_func("cora-missing-60")
def cora_config(args):
    return args


@register_func("cora-missing-80")
def cora_config(args):
    return args


@register_func("cora-missing-100")
def cora_config(args):
    return args


@register_func("citeseer-missing-0")
def citeseer_config(args):
    return args


@register_func("citeseer-missing-20")
def citeseer_config(args):
    return args


@register_func("citeseer-missing-40")
def citeseer_config(args):
    return args


@register_func("citeseer-missing-60")
def citeseer_config(args):
    return args


@register_func("citeseer-missing-80")
def citeseer_config(args):
    return args


@register_func("citeseer-missing-100")
def citeseer_config(args):
    return args


@register_func("pubmed-missing-0")
def pubmed_config(args):
    return args


@register_func("pubmed-missing-20")
def pubmed_config(args):
    return args


@register_func("pubmed-missing-40")
def pubmed_config(args):
    return args


@register_func("pubmed-missing-60")
def pubmed_config(args):
    return args


@register_func("pubmed-missing-80")
def pubmed_config(args):
    return args


@register_func("pubmed-missing-100")
def pubmed_config(args):
    return args


def run(dataset_name):
    args = build_default_args_for_node_classification(dataset_name)
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
    datasets = [
        "cora-missing-0", "cora-missing-20", "cora-missing-40",
        "cora-missing-60", "cora-missing-80", "cora-missing-100",
        "citeseer-missing-0", "citeseer-missing-20", "citeseer-missing-40",
        "citeseer-missing-60", "citeseer-missing-80", "citeseer-missing-100",
        "pubmed-missing-0", "pubmed-missing-20", "pubmed-missing-40",
        "pubmed-missing-60", "pubmed-missing-80", "pubmed-missing-100"
        ]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "sgcpn")
