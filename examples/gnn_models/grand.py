import torch

from utils import print_result, set_random_seed, get_dataset, get_extra_args
from cogdl.tasks import build_task
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
        "seed": [100],
        "input_dropout": 0.5,
        "hidden_dropout": 0.5,
        "hidden_size": 32,
        "dropnode_rate": 0.5,
        "order": 5,
        "tem": 0.5,
        "lam": 0.5,
        "sample": 10,
        "alpha": 0.2,
        "bn": False,
        "task": "node_classification",
        "model": "grand",
        "dataset": dataset,
    }
    args = get_extra_args(args)
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


@register_func("cora")
def cora_config(args):
    args.order = 8
    args.sample = 4
    args.lam = 1.0
    args.tem = 0.5
    args.alpha = 0.5
    args.patience = 200
    return args


@register_func("citeseer")
def citeseer_config(args):
    args.order = 2
    args.sample = 2
    args.lam = 0.7
    args.tem = 0.3
    args.alpha = 0.5
    args.input_dropout = 0.0
    args.hidden_dropout = 0.2
    args.patience = 200
    return args


@register_func("pubmed")
def pubmed_config(args):
    args.order = 5
    args.sample = 4
    args.lam = 1.0
    args.tem = 0.2
    args.alpha = 0.5
    args.lr = 0.2
    args.bn = True
    args.input_dropout = 0.6
    args.hidden_dropout = 0.8
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
    print_result(results, datasets, "grand")
