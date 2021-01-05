import copy
import itertools
import os
from collections import defaultdict, namedtuple

import torch
import yaml
from tabulate import tabulate

from cogdl.options import get_default_args
from cogdl.tasks import build_task
from cogdl.utils import set_random_seed, tabulate_results


def train(args):
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
    match_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "match.yml")
    with open(match_path, "r", encoding="utf8") as f:
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


def experiment(task: str, dataset, model, **kwargs):
    if "args" not in kwargs:
        args = get_default_args(task=task, dataset=dataset, model=model, **kwargs)
    else:
        args = kwargs["args"]

    variants = list(gen_variants(dataset=args.dataset, model=args.model, seed=args.seed))
    variants = check_task_dataset_model_match(task, variants)

    results_dict = defaultdict(list)
    results = [train(args) for args in variant_args_generator(args, variants)]
    for variant, result in zip(variants, results):
        results_dict[variant[:-1]].append(result)

    col_names = ["Variant"] + list(results_dict[variants[0][:-1]][-1].keys())
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=col_names, tablefmt="github"))

    return results_dict
