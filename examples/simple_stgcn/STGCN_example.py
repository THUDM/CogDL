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


# 传入参数，加载模型
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
    # 设置随机种子
    set_random_seed(args.seed)

    # 要求字符串
    model_name = args.model if isinstance(args.model, str) else args.model.model_name
    dw_name = args.dw if isinstance(args.dw, str) else args.dw.__name__
    mw_name = args.mw if isinstance(args.mw, str) else args.mw.__name__


    # 打印训练关键信息
    print(
        f""" 
|-------------------------------------{'-' * (len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|
    *** Running (`{args.dataset}`, `{model_name}`, `{dw_name}`, `{mw_name}`)
|-------------------------------------{'-' * (
                    len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|"""
    )

    # 按照指定格式 构建数据集
    """
        aa = Namespace(activation='relu', actnn=False, checkpoint_path='./checkpoints/model.pt', cpu=False, cpu_inference=False, 
    dataset=['cora'], devices=[0], distributed=False, dropout=0.5, dw='node_classification_dw', epochs=500, eval_step=1, 
    fp16=False, hidden_size=64, load_emb_path=None, local_rank=0, log_path='.', logger=None, lr=0.01, master_addr='localhost', 
    master_port=13425, max_epoch=None, model=['gcn'], mw='node_classification_mw', n_trials=3, n_warmup_steps=0, no_test=False, 
    norm=None, nstage=1, num_classes=None, num_features=None, num_layers=2, patience=100, progress_bar='epoch', 
    project='cogdl-exp', residual=False, resume_training=False, rp_ratio=1, save_emb_path=None, seed=[1], split=[0],
    unsup=False, use_best_config=False, weight_decay=0)
    """


    # 接受数据集类的实例化
    dataset = build_dataset(args)
    # cogdl.datasets.planetoid_data.CoraDataset
    # 并且 args 已经根据 数据集发生改变了

    # 获取模型封装，数据封装的类
    mw_class = fetch_model_wrapper(args.mw)
    dw_class = fetch_data_wrapper(args.dw)  # node_classification_dw

    if mw_class is None:
        raise NotImplementedError("`model wrapper(--mw)` must be specified.")

    if dw_class is None:
        raise NotImplementedError("`data wrapper(--dw)` must be specified.")

    # 定义 封装的参数
    data_wrapper_args = dict()
    model_wrapper_args = dict()


    # setup data_wrapper
    # 根据数据名称获取类的实例化
    data_wrapper_args['batch_size'] = args.batch_size
    data_wrapper_args['n_his'] = args.n_his
    data_wrapper_args['n_pred'] = args.n_pred
    data_wrapper_args['train_prop'] = args.train_prop
    data_wrapper_args['val_prop'] = args.val_prop
    data_wrapper_args['test_prop'] = args.test_prop
    data_wrapper_args['pred_length'] = args.pred_length

    dataset_wrapper = dw_class(dataset, **data_wrapper_args)
    # cogdl.wrappers.data_wrapper.node_classification.node_classification_dw.FullBatchNodeClfDataWrapper

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
    # 根据模型名称获取类的实例化
    # renxs
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
        no_test=args.no_test,
        nstage=args.nstage,
        actnn=args.actnn,
        fp16=args.fp16,
    )

    # Go!!!
    # 开始训练，传入 模型参数封装和数据集封装
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
    # 产生变量 [Variant(dataset='cora', model='gcn', seed=1, split=0)]
    variants = list(gen_variants(dataset=args.dataset, model=args.model, seed=args.seed, split=args.split))

    # defaultdict(<class 'list'>, {})
    results_dict = defaultdict(list)

    # train
    results = []
    for aa in variant_args_generator(args, variants):
        """
        aa = Namespace(activation='relu', actnn=False, checkpoint_path='./checkpoints/model.pt', cpu=False, cpu_inference=False, 
    dataset=['cora'], devices=[0], distributed=False, dropout=0.5, dw='node_classification_dw', epochs=500, eval_step=1, 
    fp16=False, hidden_size=64, load_emb_path=None, local_rank=0, log_path='.', logger=None, lr=0.01, master_addr='localhost', 
    master_port=13425, max_epoch=None, model=['gcn'], mw='node_classification_mw', n_trials=3, n_warmup_steps=0, no_test=False, 
    norm=None, nstage=1, num_classes=None, num_features=None, num_layers=2, patience=100, progress_bar='epoch', 
    project='cogdl-exp', residual=False, resume_training=False, rp_ratio=1, save_emb_path=None, seed=[1], split=[0],
    unsup=False, use_best_config=False, weight_decay=0)
        """
        results.append(train(aa))

    # results = [train(args) for args in variant_args_generator(args, variants)] # [ {'test_acc': 0.794, 'val_acc': 0.788} ]
    # list(zip(variants, results)) [(Variant(dataset='cora', model='gcn', seed=1, split=0), {'test_acc': 0.794, 'val_acc': 0.788})]
    for variant, result in zip(variants, results):
        # 取('cora', 'gcn') ：[ {'test_acc': 0.794, 'val_acc': 0.788} ]
        results_dict[variant[:-2]].append(result)

    tablefmt = "github"
    output_results(results_dict, tablefmt)

    return results_dict


def experiment(dataset, model=None, **kwargs):
    dataset = [dataset]
    model = [model]
    #  获取 Namespace 参数空间
    # 从 wrappers 获取初始化的参数, 继承基础参数设置，并 根据需要添加新的参数
    args = get_default_args(dataset=[str(x) for x in dataset], model=[str(x) for x in model], **kwargs)
    args.dataset = dataset
    args.model = model
    return raw_experiment(args)

def files_exist(files):
    # 返回一个布尔变量
    return all([osp.exists(f) for f in files])


if __name__ == "__main__":

    kwargs = {"epochs":1,
              "kernel_size":3,
              "n_his":20,
              "n_pred":1,
              "channel_size_list":np.array([[ 1, 16, 64],[64, 64, 64],[64, 16, 64]]),
              "num_layers":3,
              "num_nodes":288,
              "train_prop": 0.8,
              "val_prop": 0.1,
              "test_prop": 0.1,
              "pred_length":288,}

    experiment(dataset="pems-stgat", model="stgat", resume_training=False, **kwargs)
