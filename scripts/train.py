import copy
import itertools
import random
from numpy.core.fromnumeric import product
import yaml
from collections import defaultdict, namedtuple

import numpy as np
import torch
from tabulate import tabulate

from cogdl import options
from cogdl.tasks import build_task
from cogdl.utils import set_random_seed, tabulate_results



def main(args):
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id[0])

    set_random_seed(args.seed)

    task = build_task(args)
    result = task.train()
    if "ValAcc" in result:
        result.pop("ValAcc")
    return result


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


def variant_args_generator(args, variants):
    """Form variants as group with size of num_workers"""
    for variant in variants:
        args.dataset, args.model, args.seed = variant
        yield copy.deepcopy(args)


def check_task_dataset_model_match(task, variants):
    with open("./match.yml", "r", encoding="utf8") as f:
        match = yaml.load(f)
    objective = match.get(task, None)
    if objective is None:
        raise NotImplementedError
    pairs = []
    for item in objective:
        pairs.extend([(x, y) for x in item["model"] for y in item["dataset"]])

    clean_variants = []
    for item in variants:
        if (item.model, item.dataset) not in pairs:
            print(f"({item.model}, {item.dataset}) is not implemented in task '{task}''.")
            continue
        clean_variants.append(item)
    if not clean_variants:
        exit(0)
    return clean_variants


if __name__ == "__main__":
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)
    assert len(args.device_id) == 1
    variants = list(
        gen_variants(dataset=args.dataset, model=args.model, seed=args.seed)
    )
    variants = check_task_dataset_model_match(args.task, variants)
    # Collect results
    results_dict = defaultdict(list)
    results = [main(args) for args in variant_args_generator(args, variants)]
    for variant, result in zip(variants, results):
        results_dict[variant[:-1]].append(result)

    col_names = ["Variant"] + list(results_dict[variant[:-1]][-1].keys())
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=col_names, tablefmt="github"))
