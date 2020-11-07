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
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epoch": 1000,
        "patience": 100,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0, 1, 2],

        "dropout": 0.7,
        "layer1_pows": [200, 200, 200],
        "layer2_pows": [20, 20, 20],

        "task": "node_classification",
        "model": "mixhop",
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
    datasets = ["cora", "citeseer", "pubmed"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "mixhop")
