from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict, print_result, set_random_seed
import torch
DATASET_REGISTRY = {}


def build_default_args_for_node_classification(dataset):
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.001,
        "cpu": cpu,
        "device_id": [0],
        "weight_decay": 0.001,
        "max_epoch": 2000,
        "patience": 20,
        "seed": [0],
        "dropout": 0.5,
        "hidden_size": 64,
        "niter": 10,
        "prop_penalty": 0.005,
        "missing_rate": -1,
        "task": "node_classification",
        "model": "ap_gcn",
        "dataset": dataset,
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func('citeseer')
def citeseer_config(args):
    return args


def run(dataset_name):
    args = build_default_args_for_node_classification(dataset_name)
    args = DATASET_REGISTRY[dataset_name](args)
    dataset = build_dataset(args)
    results = []
    for seed in args.seed:
        set_random_seed(seed)
        task = build_task(args, dataset=dataset)
        result = task.train()
        results.append(result)
    return results


if __name__ == "__main__":
    datasets = ['citeseer']
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "ap_gcn")
