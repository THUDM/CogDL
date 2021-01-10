import copy
import itertools
import os
from collections import defaultdict, namedtuple

import torch
import yaml
import optuna
from tabulate import tabulate

from cogdl.options import get_default_args
from cogdl.tasks import build_task
from cogdl.utils import set_random_seed, tabulate_results


class AutoML(object):
    """
    Args:
        func_search: function to obtain hyper-parameters to search
    """

    def __init__(self, task, dataset, model, n_trials=3, **kwargs):
        self.task = task
        self.dataset = dataset
        self.model = model
        self.seed = kwargs.pop("seed") if "seed" in kwargs else [1]
        assert "func_search" in kwargs
        self.func_search = kwargs["func_search"]
        self.n_trials = n_trials
        self.best_value = None
        self.default_params = kwargs

    def _objective(self, trials):
        params = self.default_params
        params.update(self.func_search(trials))
        result_dict = raw_experiment(task=self.task, dataset=self.dataset, model=self.model, seed=self.seed, **params)
        result_list = list(result_dict.values())[0]
        item = result_list[0]
        key = None
        for _key in item.keys():
            if "Val" in _key or "val" in _key:
                key = _key
                break
        if key is None:
            raise KeyError("Unable to find validation metrics")
        val = [result[key] for result in result_list]
        mean = sum(val) / len(val)

        if self.best_value is None or mean > self.best_value:
            self.best_value = mean
            self.best_results = result_list

        return mean

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)

        return self.best_results


def train(args):
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id[0])

    set_random_seed(args.seed)

    task = build_task(args)
    result = task.train()

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


def output_results(results_dict, tablefmt="github"):
    variant = list(results_dict.keys())[0]
    col_names = ["Variant"] + list(results_dict[variant][-1].keys())
    tab_data = tabulate_results(results_dict)

    print(tabulate(tab_data, headers=col_names, tablefmt=tablefmt))


def raw_experiment(task: str, dataset, model, **kwargs):
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

    tablefmt = kwargs["tablefmt"] if "tablefmt" in kwargs else "github"
    output_results(results_dict, tablefmt)

    return results_dict


def auto_experiment(task: str, dataset, model, **kwargs):
    variants = list(gen_variants(dataset=dataset, model=model))
    variants = check_task_dataset_model_match(task, variants)

    results_dict = defaultdict(list)
    for variant in variants:
        tool = AutoML(task, variant.dataset, variant.model, **kwargs)
        results_dict[variant[:]] = tool.run()

    tablefmt = kwargs["tablefmt"] if "tablefmt" in kwargs else "github"
    print("\nFinal results:\n")
    output_results(results_dict, tablefmt)

    return results_dict


def experiment(task: str, dataset, model, **kwargs):
    if "func_search" in kwargs:
        if isinstance(dataset, str):
            dataset = [dataset]
        if isinstance(model, str):
            model = [model]
        return auto_experiment(task, dataset, model, **kwargs)

    return raw_experiment(task, dataset, model, **kwargs)
