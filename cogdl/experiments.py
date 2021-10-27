import copy
import itertools
import os
import inspect
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import optuna
from tabulate import tabulate

from cogdl.utils import set_random_seed, tabulate_results, init_operator_configs
from cogdl.configs import BEST_CONFIGS
from cogdl.data import Dataset
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
from cogdl.options import get_default_args
from cogdl.trainer import Trainer


class AutoML(object):
    """
    Args:
        search_space: function to obtain hyper-parameters to search
    """

    def __init__(self, args):
        self.search_space = args.search_space
        self.metric = args.metric if hasattr(args, "metric") else None
        self.n_trials = args.n_trials if hasattr(args, "n_trials") else 3
        self.best_value = None
        self.best_params = None
        self.default_params = args

    def _objective(self, trials):
        params = copy.deepcopy(self.default_params)
        cur_params = self.search_space(trials)
        print(cur_params)
        for key, value in cur_params.items():
            params.__setattr__(key, value)
        result_dict = raw_experiment(args=params)
        result_list = list(result_dict.values())[0]
        item = result_list[0]
        key = self.metric
        if key is None:
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
            self.best_params = cur_params
            self.best_results = result_list

        return mean

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        print(study.best_params)
        return self.best_results


def examine_link_prediction(args, dataset):
    if "link_prediction" in args.mw:
        args.num_entities = dataset.data.num_nodes
        # args.num_entities = len(torch.unique(self.data.edge_index))
        if dataset.data.edge_attr is not None:
            args.num_rels = len(torch.unique(dataset.data.edge_attr))
            args.monitor = "mrr"
        else:
            args.monitor = "auc"
    return args


def set_best_config(args):
    configs = BEST_CONFIGS[args.task]
    if args.model not in configs:
        return args
    configs = configs[args.model]
    for key, value in configs["general"].items():
        args.__setattr__(key, value)
    if args.dataset not in configs:
        return args
    for key, value in configs[args.dataset].items():
        args.__setattr__(key, value)
    return args


def train(args):  # noqa: C901
    if isinstance(args.dataset, list):
        args.dataset = args.dataset[0]
    if isinstance(args.model, list):
        args.model = args.model[0]
    if isinstance(args.seed, list):
        args.seed = args.seed[0]
    set_random_seed(args.seed)

    model_name = args.model if isinstance(args.model, str) else args.model.model_name

    print(
        f""" 
|-------------------------------------{'-' * (len(str(args.dataset)) + len(model_name) + len(args.dw) + len(args.mw))}|
    *** Running (`{args.dataset}`, `{model_name}`, `{args.dw}`, `{args.mw}`)
|-------------------------------------{'-' * (len(str(args.dataset)) + len(model_name) + len(args.dw) + len(args.mw))}|"""
    )

    if getattr(args, "use_best_config", False):
        args = set_best_config(args)

    # setup dataset and specify `num_features` and `num_classes` for model
    args.monitor = "val_acc"
    if isinstance(args.dataset, Dataset):
        dataset = args.dataset
    else:
        dataset = build_dataset(args)

    mw_class = fetch_model_wrapper(args.mw)
    dw_class = fetch_data_wrapper(args.dw)

    if mw_class is None:
        raise NotImplementedError("`model wrapper(--mw)` must be specified.")

    if dw_class is None:
        raise NotImplementedError("`data wrapper(--dw)` must be specified.")

    data_wrapper_args = dict()
    model_wrapper_args = dict()
    # unworthy code: share `args` between model and dataset_wrapper
    for key in inspect.signature(dw_class).parameters.keys():
        if hasattr(args, key) and key != "dataset":
            data_wrapper_args[key] = getattr(args, key)
    # unworthy code: share `args` between model and model_wrapper
    for key in inspect.signature(mw_class).parameters.keys():
        if hasattr(args, key) and key != "model":
            model_wrapper_args[key] = getattr(args, key)

    args = examine_link_prediction(args, dataset)

    # setup data_wrapper
    dataset_wrapper = dw_class(dataset, **data_wrapper_args)

    args.num_features = dataset.num_features
    if hasattr(dataset, "num_nodes"):
        args.num_nodes = dataset.num_nodes
    if hasattr(dataset, "num_edges"):
        args.num_edges = dataset.num_edges
    if hasattr(dataset, "num_edge"):
        args.num_edge = dataset.num_edge
    if hasattr(dataset, "max_graph_size"):
        args.max_graph_size = dataset.max_graph_size
    if hasattr(dataset, "edge_attr_size"):
        args.edge_attr_size = dataset.edge_attr_size
    else:
        args.edge_attr_size = [0]
    if hasattr(args, "unsup") and args.unsup:
        args.num_classes = args.hidden_size
    else:
        args.num_classes = dataset.num_classes
    if hasattr(dataset.data, "edge_attr") and dataset.data.edge_attr is not None:
        args.num_entities = len(torch.unique(torch.stack(dataset.data.edge_index)))
        args.num_rels = len(torch.unique(dataset.data.edge_attr))

    # setup model
    if isinstance(args.model, nn.Module):
        model = args.model
    else:
        model = build_model(args)
    # specify configs for optimizer
    optimizer_cfg = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_warmup_steps=args.n_warmup_steps,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 0,
    )

    if hasattr(args, "hidden_size"):
        optimizer_cfg["hidden_size"] = args.hidden_size

    # setup model_wrapper
    if "embedding" in args.mw:
        model_wrapper = mw_class(model, **model_wrapper_args)
    else:
        model_wrapper = mw_class(model, optimizer_cfg, **model_wrapper_args)

    os.makedirs("./checkpoints", exist_ok=True)

    # setup controller
    trainer = Trainer(
        max_epoch=args.max_epoch,
        device_ids=args.devices,
        cpu=args.cpu,
        save_emb_path=args.save_emb_path,
        load_emb_path=args.load_emb_path,
        cpu_inference=args.cpu_inference,
        # monitor=args.monitor,
        progress_bar=args.progress_bar,
        distributed_training=args.distributed,
        checkpoint_path=args.checkpoint_path,
        resume_training=args.resume_training,
        patience=args.patience,
        logger=args.logger,
        log_path=args.log_path,
        project=args.project,
        no_test=args.no_test,
        nstage=args.nstage,
        actnn=args.actnn,
    )

    # Go!!!
    result = trainer.run(model_wrapper, dataset_wrapper)

    return result


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


def variant_args_generator(args, variants):
    """Form variants as group with size of num_workers"""
    for variant in variants:
        args.dataset, args.model, args.seed = variant
        yield copy.deepcopy(args)


def output_results(results_dict, tablefmt="github"):
    variant = list(results_dict.keys())[0]
    col_names = ["Variant"] + list(results_dict[variant][-1].keys())
    tab_data = tabulate_results(results_dict)

    print(tabulate(tab_data, headers=col_names, tablefmt=tablefmt))


def raw_experiment(args):
    init_operator_configs(args)

    variants = list(gen_variants(dataset=args.dataset, model=args.model, seed=args.seed))

    results_dict = defaultdict(list)
    results = [train(args) for args in variant_args_generator(args, variants)]
    for variant, result in zip(variants, results):
        results_dict[variant[:-1]].append(result)

    tablefmt = args.tablefmt if hasattr(args, "tablefmt") else "github"
    output_results(results_dict, tablefmt)

    return results_dict


def auto_experiment(args):
    variants = list(gen_variants(dataset=args.dataset, model=args.model))

    results_dict = defaultdict(list)
    for variant in variants:
        args.model = [variant.model]
        args.dataset = [variant.dataset]
        tool = AutoML(args)
        results_dict[variant[:]] = tool.run()

    tablefmt = args.tablefmt if hasattr(args, "tablefmt") else "github"
    print("\nFinal results:\n")
    output_results(results_dict, tablefmt)

    return results_dict


def experiment(dataset, model, **kwargs):
    if isinstance(dataset, str) or isinstance(dataset, Dataset):
        dataset = [dataset]
    if isinstance(model, str) or isinstance(model, nn.Module):
        model = [model]
    if "args" not in kwargs:
        args = get_default_args(dataset=[str(x) for x in dataset], model=[str(x) for x in model], **kwargs)
    else:
        args = kwargs["args"]
        for key, value in kwargs.items():
            if key != "args":
                args.__setattr__(key, value)
    args.dataset = dataset
    args.model = model

    if "search_space" in kwargs:
        return auto_experiment(args)

    return raw_experiment(args)
