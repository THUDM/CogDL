import sys,os
add_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(add_path.split('examples')[0])
print(add_path.split('examples')[0])

import os.path as osp
from cogdl import experiment
import time
import copy
import itertools
import os
import inspect
from collections import defaultdict, namedtuple
import warnings
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import optuna
from tabulate import tabulate
from cogdl.utils import set_random_seed, tabulate_results, download_url, makedirs, untar
from cogdl.configs import BEST_CONFIGS
from cogdl.data import Dataset
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
from cogdl.options import get_default_args
from cogdl.trainer import Trainer
import numpy as np




def output_results(results_dict, tablefmt="github"):
    variant = list(results_dict.keys())[0]
    col_names = ["Variant"] + list(results_dict[variant][-1].keys())
    tab_data = tabulate_results(results_dict)

    print(tabulate(tab_data, headers=col_names, tablefmt=tablefmt))


def train(args):  # noqa: C901
    if isinstance(args.dataset, list):
        args.dataset = args.dataset[0]
    if isinstance(args.model, list):
        args.model = args.model[0]
    if isinstance(args.seed, list):
        args.seed = args.seed[0]
    if isinstance(args.split, list):
        args.split = args.split[0]
    # dataset='cora', model='gcn', seed=1, split=0
    set_random_seed(args.seed)


    model_name = args.model if isinstance(args.model, str) else args.model.model_name
    dw_name = args.dw if isinstance(args.dw, str) else args.dw.__name__
    mw_name = args.mw if isinstance(args.mw, str) else args.mw.__name__

    print(
        f""" 
|-------------------------------------{'-' * (len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|
    *** Running (`{args.dataset}`, `{model_name}`, `{dw_name}`, `{mw_name}`)
|-------------------------------------{'-' * (
                    len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|"""
    )


    dataset = build_dataset(args)
    mw_class = fetch_model_wrapper(args.mw)
    dw_class = fetch_data_wrapper(args.dw)

    if mw_class is None:
        raise NotImplementedError("`model wrapper(--mw)` must be specified.")

    if dw_class is None:
        raise NotImplementedError("`data wrapper(--dw)` must be specified.")

    data_wrapper_args = dict()
    model_wrapper_args = dict()

    data_wrapper_args['batch_size'] = args.batch_size
    data_wrapper_args['n_his'] = args.n_his
    data_wrapper_args['n_pred'] = args.n_pred
    data_wrapper_args['train_prop'] = args.train_prop
    data_wrapper_args['val_prop'] = args.val_prop
    data_wrapper_args['test_prop'] = args.test_prop
    data_wrapper_args['pred_length'] = args.pred_length

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
        epochs=args.epochs,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 0,
    )

    if hasattr(args, "hidden_size"):
        optimizer_cfg["hidden_size"] = args.hidden_size

    # setup model_wrapper
    model_wrapper_args['edge_index'] = dataset.data.edge_index
    model_wrapper_args['edge_weight'] = dataset.data.edge_weight
    model_wrapper_args['scaler'] = dataset_wrapper.scaler
    model_wrapper_args['node_ids'] = dataset.data.node_ids
    model_wrapper_args['pred_timestamp'] = dataset_wrapper.get_pre_timestamp()
    if isinstance(args.mw, str) and "embedding" in args.mw:
        model_wrapper = mw_class(model, **model_wrapper_args)
    else:
        model_wrapper = mw_class(model, optimizer_cfg, **model_wrapper_args)

    os.makedirs("./checkpoints", exist_ok=True)

    # setup controller
    trainer = Trainer(
        epochs=args.epochs,
        device_ids=args.devices,
        cpu=args.cpu,
        save_emb_path=args.save_emb_path,
        load_emb_path=args.load_emb_path,
        cpu_inference=args.cpu_inference,
        progress_bar=args.progress_bar,
        distributed_training=args.distributed,
        checkpoint_path=args.checkpoint_path,
        resume_training=args.resume_training,
        patience=args.patience,
        eval_step=args.eval_step,
        logger=args.logger,
        log_path=args.log_path,
        project=args.project,
        nstage=args.nstage,
        actnn=args.actnn,
        fp16=args.fp16,
    )

    result = trainer.run(model_wrapper, dataset_wrapper)

    return result


def variant_args_generator(args, variants):
    """Form variants as group with size of num_workers"""
    for idx, variant in enumerate(variants):
        args.dataset, args.model, args.seed, args.split = variant
        yield copy.deepcopy(args)


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


def raw_experiment(args):
    variants = list(gen_variants(dataset=args.dataset, model=args.model, seed=args.seed, split=args.split))
    results_dict = defaultdict(list)

    # train
    results = []
    for aa in variant_args_generator(args, variants):
        results.append(train(aa))
    for variant, result in zip(variants, results):
        results_dict[variant[:-2]].append(result)
    tablefmt = "github"
    output_results(results_dict, tablefmt)

    return results_dict


def experiment(dataset, model=None, **kwargs):
    dataset = [dataset]
    model = [model]
    args = get_default_args(dataset=[str(x) for x in dataset], model=[str(x) for x in model], **kwargs)
    args.dataset = dataset
    args.model = model
    return raw_experiment(args)

def files_exist(files):
    return all([osp.exists(f) for f in files])


if __name__ == "__main__":

    kwargs = {"epochs":1,
              "kernel_size":3,
              "n_his":20,
              "n_pred":1,
              "channel_size_list":np.array([[ 1, 4, 8],[8, 8, 8],[8, 4, 8]]),
              "num_layers":3,
              "num_nodes":288,
              "train_prop": 0.1,
              "val_prop": 0.1,
              "test_prop": 0.1,
              "pred_length":288,}

    # experiment(dataset="pems-stgcn", model="stgcn", resume_training=False, **kwargs)
    experiment(dataset="pems-stgat", model="stgat", resume_training=False, **kwargs)
