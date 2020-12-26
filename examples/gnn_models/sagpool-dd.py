import random
import numpy as np

import torch

from utils import print_result, set_random_seed, get_dataset
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

DATASET_REGISTRY = {}

def build_default_args_for_graph_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.0005,
        "weight_decay": 0.0001,
        "max_epoch": 100000,
        "patience": 50,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0],
        "train_ratio": 0.8,
        "test_ratio": 0.1,
        "batch_size": 128,
        "kfold": False,
        "degree_feature": False,
        "uniform_feature": False,
        "gamma": 0.5,
        "dropout": 0.5,
        "hidden_size": 128,
        "pooling_ratio": 0.5,
        "pooling_layer_type": "gcnconv",
        "task": "graph_classification",
        "model": "sagpool",
        "dataset": dataset,
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func("dd")
def dd_config(args):
    return args


def run(dataset_name):
    args = build_default_args_for_graph_classification(dataset_name)
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
    datasets = ["dd"]
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "sagpool")
