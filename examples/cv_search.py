import copy
import datetime
import torch
import time
from itertools import product

from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict, set_random_seed


def get_time():
    return datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S")


def get_dataset(args):
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    return dataset, args


def build_default_args():
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.01,
        "weight_decay": 0.001,
        "max_epoch": 1000,
        "patience": 100,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0, 1, 2],
        "task": "node_classification",
        "model": "unet",
        "dataset": "cora",
        "n_pool": 4,
        "pool_rate": [0.7, 0.5, 0.5, 0.4],
        "missing_rate": -1,
    }
    return build_args_from_dict(args)


def get_parameters():
    scope = {
        "lr": [0.01, 0.001, 0.0001],
        "n_dropout": [0.5, 0.7, 0.9],
        "adj_dropout": [0.0, 0.2, 0.4],
        "hidden_size": [64, 128],
        "activation": ["relu", "identity"],
        "improved": [True, False],
        "aug_adj": [True, False],
    }
    keys = list(scope.keys())
    values = list(scope.values())
    combination = product(*values)
    return [{keys[i]: val[i] for i in range(len(keys))} for val in combination]


def train():
    args = build_default_args()
    dataset, args = get_dataset(args)

    combinations = get_parameters()
    best_parameters = None
    best_result = None
    best_val_acc = 0

    print(f"===== Start At: {get_time()} ===========")
    start = time.time()

    random_seeds = list(range(5))
    for item in combinations:
            for key, val in item.items():
                setattr(args, key, val)

            print(f"### -- Parameters: {args.__dict__}")
            result_list = []
            for seed in random_seeds:
                set_random_seed(seed)

                task = build_task(args, dataset=dataset)
                res = task.train()
                result_list.append(res)

            val_acc = [x["ValAcc"] for x in result_list]
            test_acc = [x["Acc"] for x in result_list]
            val_acc = sum(val_acc) / len(val_acc)
            print(f"###    Result: {val_acc}")
            if val_acc > best_val_acc:
                best_parameters = copy.deepcopy(args)
                best_result = dict(Acc=sum(test_acc)/len(test_acc), ValAcc=val_acc)
    print(f"Best Parameters: {best_parameters}")
    print(f"Best result: {best_result}")

    end = time.time()
    print(f"===== End At: {get_time()} ===========")
    print("Time cost:", end - start)


if __name__ == "__main__":
    train()
