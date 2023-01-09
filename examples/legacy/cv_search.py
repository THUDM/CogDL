import copy
import datetime
import torch
import time
from itertools import product

from cogdl import experiment
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict, set_random_seed


def get_time():
    return datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S")


def build_default_args():
    cpu = not torch.cuda.is_available()
    args = {
        "lr": 0.01,
        "weight_decay": 0.001,
        "epochs": 1000,
        "patience": 100,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0, 1],
    }
    return args


def get_parameters():
    scope = {
        "lr": [0.01, 0.001],
        "dropout": [0.5, 0.7],
        "hidden_size": [64, 128],
    }
    keys = list(scope.keys())
    values = list(scope.values())
    combination = product(*values)
    return [{keys[i]: val[i] for i in range(len(keys))} for val in combination]


def train():
    args = build_default_args()

    combinations = get_parameters()
    best_parameters = None
    best_result = None
    best_val_acc = 0

    print(f"===== Start At: {get_time()} ===========")
    start = time.time()

    for item in combinations:
        for key, val in item.items():
            args[key] = val

        print(f"### -- Parameters: {args}")

        res = experiment(dataset="cora", model="gcn", **args)
        result_list = list(res.values())[0]

        val_acc = [x["val_acc"] for x in result_list]
        test_acc = [x["test_acc"] for x in result_list]
        val_acc = sum(val_acc) / len(val_acc)
        print(f"###    Result: {val_acc}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_parameters = copy.deepcopy(args)
            best_result = dict(Acc=sum(test_acc) / len(test_acc), ValAcc=val_acc)
    print(f"Best Parameters: {best_parameters}")
    print(f"Best result: {best_result}")

    end = time.time()
    print(f"===== End At: {get_time()} ===========")
    print("Time cost:", end - start)


if __name__ == "__main__":
    train()
