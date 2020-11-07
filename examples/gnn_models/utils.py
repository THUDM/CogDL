import itertools
import random
import torch
import numpy as np
from collections import defaultdict

from tabulate import tabulate
from cogdl.datasets import build_dataset


def tabulate_results(results_dict):
    # Average for different seeds
    tab_data = []
    for variant in results_dict:
        results = np.array([list(res.values()) for res in results_dict[variant]])
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(
                        np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist(),
                    ),
                )
            )
        )
    return tab_data


def print_result(results, datasets, model_name):
    table_header = ["Variants"] + list(results[0].keys())

    results_dict = defaultdict(list)
    num_datasets = len(datasets)
    num_seed = len(results) // num_datasets
    for i, res in enumerate(results):
        results_dict[(model_name, datasets[i//num_seed])].append(res)
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=table_header, tablefmt="github"))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_dataset(args):
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    return dataset, args
