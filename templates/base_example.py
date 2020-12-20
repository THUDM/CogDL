from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict, print_result, set_random_seed

DATASET_REGISTRY = {}


def build_default_args_for_unsupervised_node_classification(dataset):
    args = {
        "lr": 0.001,
        "weight_decay": 0,
        "max_epoch": 1000,
        "max_epochs": 1000,
        "patience": 20,
        "device_id": [0],
        "seed": [0],
        "dropout": 0.0,
        "hidden_size": 512,
        "num_layers": 2,
        "task": "model_task",
        "model": "model_name",
        "dataset": dataset,
    }
    return build_args_from_dict(args)


def register_func(name):
    def register_func_name(func):
        DATASET_REGISTRY[name] = func
        return func

    return register_func_name


def run(dataset_name):
    args = build_default_args_for_unsupervised_node_classification(dataset_name)
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
    datasets = []
    results = []
    for x in datasets:
        results += run(x)
    print_result(results, datasets, "model_name")
