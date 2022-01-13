import copy
import itertools
import json
import os
import random
import string
import inspect
from collections import defaultdict, namedtuple
import warnings

import torch
import torch.nn as nn
import optuna
from tabulate import tabulate

from cogdl.utils import set_random_seed, tabulate_results
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

        self.trial_log = {}
        self.trial_cnt = 0
        self.trial_history = {}

    def _objective(self, trials):
        try:
            params = copy.deepcopy(self.default_params)
            cur_params = self.search_space(trials)
            print(cur_params)
            
            tag_params = str(cur_params)
            if tag_params not in self.trial_history:
                for key, value in cur_params.items():
                    params.__setattr__(key, value)
                result_dict = raw_experiment(args=params)
                result_list = list(result_dict.values())[0]
                item = result_list[0]
                
                result_mean = {}
                # Mean
                for _key in item.keys():
                    val = [result[_key] for result in result_list]
                    mean = sum(val) / len(val)
                    result_mean[_key] = mean
                
                # Std
                for _key in item.keys():
                    mean_val = result_mean[_key]
                    val = [(result[_key] - mean_val) ** 2.0 for result in result_list]
                    mean = (sum(val) / len(val)) ** 0.5
                    result_mean[_key+'_std'] = mean
                self.trial_history[tag_params] = copy.deepcopy(result_mean)
            else:
                result_mean = self.trial_history[tag_params]
                
            key = self.metric
            if key is None:
                for _key in result_mean.keys():
                    if "Val" in _key or "val" in _key:
                        key = _key
                        break
                if key is None:
                    raise KeyError("Unable to find validation metrics")
            mean = result_mean[key]
            
            cur_params.update(result_mean)
            if self.best_value is None or mean > self.best_value:
                self.best_value = mean
                self.best_params = cur_params
                self.best_results = result_list
                self.best_mean = result_mean
            
            self.trial_log['Trial{:_>4d}'.format(self.trial_cnt)] = copy.deepcopy(cur_params)
            self.trial_cnt += 1
            print (result_mean)
            
            return mean
        except Exception as e:
            print ('Exception in _objective', str(e))
            return 0.0

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        # fig1 = optuna.visualization.plot_optimization_history(study)
        # fig1.show()
        # fig2 = optuna.visualization.plot_slice(study)
        # fig2.show()
        # fig3 = optuna.visualization.plot_param_importances(study)
        # fig3.show()
        print(study.best_params)
        self.trial_log['Trial_best'] = self.best_params
        return self.best_results, self.trial_log

def set_best_config(args):
    path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'configs.json'
    with open(path, 'r') as file:
        BEST_CONFIGS = json.load(file)
    if args.model not in BEST_CONFIGS:
        return args
    configs = BEST_CONFIGS[args.model]
    for key, value in configs["general"].items():
        if '_acc' in key:
            continue
        args.__setattr__(key, value)
    if args.dataset not in configs:
        return args
    for key, value in configs[args.dataset].items():
        if '_acc' in key:
            continue
        args.__setattr__(key, value)
    return args


def train(args):  # noqa: C901
    if isinstance(args.dataset, list):
        args.dataset = args.dataset[0]
    if isinstance(args.model, list):
        args.model = args.model[0]
    if isinstance(args.seed, list):
        args.seed = args.seed[0]
    if isinstance(args.split, list):
        args.split = args.split[0]
    set_random_seed(args.seed)

    model_name = args.model if isinstance(args.model, str) else args.model.model_name
    dw_name = args.dw if isinstance(args.dw, str) else args.dw.__name__
    mw_name = args.mw if isinstance(args.mw, str) else args.mw.__name__

    print(
        f""" 
|-------------------------------------{'-' * (len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|
    *** Running (`{args.dataset}`, `{model_name}`, `{dw_name}`, `{mw_name}`)
|-------------------------------------{'-' * (len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|"""
    )

    if getattr(args, "use_best_config", False):
        args = set_best_config(args)

    # setup dataset and specify `num_features` and `num_classes` for model
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
        epochs=args.epochs,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 0,
    )

    if hasattr(args, "hidden_size"):
        optimizer_cfg["hidden_size"] = args.hidden_size

    # setup model_wrapper
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
    result = trainer.run(model_wrapper, dataset_wrapper)

    return result


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


def variant_args_generator(args, variants):
    """Form variants as group with size of num_workers"""
    for idx, variant in enumerate(variants):
        args.dataset, args.model, args.seed, args.split = variant
        yield copy.deepcopy(args)


def output_results(results_dict, tablefmt="github"):
    variant = list(results_dict.keys())[0]
    col_names = ["Variant"] + list(results_dict[variant][-1].keys())
    tab_data = tabulate_results(results_dict)

    print(tabulate(tab_data, headers=col_names, tablefmt=tablefmt))


def raw_experiment(args):
    random.seed()
    if args.checkpoint_path == './checkpoints/model.pt':
        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        args.checkpoint_path = args.checkpoint_path[:-3] + '_' + ran_str + '.pt'
        
    variants = list(gen_variants(dataset=args.dataset, model=args.model, seed=args.seed, split=args.split))

    results_dict = defaultdict(list)
    results = [train(args) for args in variant_args_generator(args, variants)]
    for variant, result in zip(variants, results):
        results_dict[variant[:-2]].append(result)

    tablefmt = args.tablefmt if hasattr(args, "tablefmt") else "github"
    output_results(results_dict, tablefmt)

    return results_dict


def auto_experiment(args):
    variants = list(gen_variants(dataset=args.dataset, model=args.model))
    
    results_dict = defaultdict(list)
    trials_dict = defaultdict(list)
    for variant in variants:
        try:
            args.model = [variant.model]
            args.dataset = [variant.dataset]
            
            if isinstance(variant.dataset, Dataset):
                dataset_name = str(variant.dataset)
            else:
                dataset_name = variant.dataset
            if isinstance(variant.model, nn.Module):
                model_name = variant.model.model_name
            else:
                model_name = variant.model
            random.seed()
            ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            args.checkpoint_path= './checkpoints/%s_%s_%s_device_%d_%s.pt' % (args.project, model_name, dataset_name, args.devices[0], ran_str)
            tool = AutoML(args)
            best_results, trial_logs = tool.run()
            if isinstance(variant[1], nn.Module):
                tmp_model_name = variant[1].model_name
            else:
                tmp_model_name = str(variant[1])
            variant_name = '(%s, %s)' % (str(variant[0]), tmp_model_name)
            results_dict[variant_name] = best_results
            trials_dict[variant_name] = trial_logs
            
        except Exception as e:
            print (f'Exception at Model {variant.model}, Dataset {variant.dataset}, Exception {str(e)}')

    tablefmt = args.tablefmt if hasattr(args, "tablefmt") else "github"
    print("\nFinal results:\n")
    output_results(results_dict, tablefmt)

    return (results_dict, trials_dict)


def default_search_space(trial):
    return {
        "dropout": trial.suggest_uniform("dropout", 0.2, 0.6),  # intra-layer
        "norm": trial.suggest_categorical("norm", ["batchnorm", "layernorm"]),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "layers_type": trial.suggest_categorical("layers_type", ["gcn", "gat", "grand", "gcnii", "drgat"]),
        "residual": trial.suggest_categorical("residual", [True, False]),  # inter-layer
        "num_layers": trial.suggest_categorical("num_layers", [2, 4, 8]),
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),  # config
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        # "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam"]),
        # "epochs": trial.suggest_categorical("epochs", [500, 1000, 1500]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4]),
    }

def check_experiment(dataset, model, use_best_config=False, args=None, devices=[0]):
    seed = list(range(20))
    if dataset in ["cora_geom", "citeseer_geom", "pubmed_geom", "chameleon", "squirrel", "film", "cornell", "texas", "wisconsin"]:
        split = list(range(10))
    else:
        split = list(range(1))
    
    if use_best_config:
        result_dict = experiment(dataset=dataset, model=model, split=split, seed=seed, use_best_config=True, devices=devices)
    else:
        result_dict = experiment(dataset=dataset, model=model, split=split, seed=seed, devices=devices, args=args)
    
    result_list = list(result_dict.values())[0]
    item = result_list[0]
    
    result_mean = {}
    # Mean
    for _key in item.keys():
        val = [result[_key] for result in result_list]
        mean = sum(val) / len(val)
        result_mean[_key+'_mean'] = mean
    
    # Std
    for _key in item.keys():
        mean_val = result_mean[_key+'_mean']
        val = [(result[_key] - mean_val) ** 2.0 for result in result_list]
        mean = (sum(val) / len(val)) ** 0.5
        result_mean[_key+'_std'] = mean

    return result_mean

def result_update(trial_log_dict, devices):
    for key, value in trial_log_dict.items():
        dataset = key.split(',')[0].split('(')[1].strip()
        model = key.split(',')[1].split(')')[0].strip()
        kwargs = copy.deepcopy(value['Trial_best'])
        for _k in value['Trial_best']:
            if '_acc' in _k:
                del kwargs[_k]

        args = get_default_args(dataset=[dataset], model=[model], **kwargs)

        result_mean = check_experiment(dataset=dataset, model=model, args=args, devices=devices)

        val_acc_mean = result_mean['val_acc_mean']
        val_acc_std = result_mean['val_acc_std']
        test_acc_mean = result_mean["test_acc_mean"]
        test_acc_std = result_mean['test_acc_std']

        path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'configs.json'
        with open(path, 'r') as file:
            configuration = json.load(file)
        
        old_result= configuration.get(model, {}).get(dataset, {})
        old_val_acc_mean = old_result.get('val_acc_mean', 0.0)
        old_val_acc_std = old_result.get('val_acc_std', 0.0)
        old_test_acc_mean = old_result.get("test_acc_mean", 0.0)
        old_test_acc_std = old_result.get('test_acc_std', 0.0)

        if val_acc_mean > old_val_acc_mean:
            if model not in configuration:
                configuration[model] = {}
            
            kwargs['val_acc_mean'] = val_acc_mean
            kwargs['val_acc_std'] = val_acc_std
            kwargs["test_acc_mean"] = test_acc_mean
            kwargs["test_acc_std"] = test_acc_std
            configuration[model][dataset] = copy.deepcopy(kwargs)

            path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'configs.json'
            with open(path, 'w') as file:
                json.dump(configuration, file, indent=4, ensure_ascii=False)
            
            print ("Update configs of (%s, %s) successd: %5.2f±%4.2f / %5.2f±%4.2f -> %5.2f±%4.2f / %5.2f±%4.2f " % (model, dataset, old_test_acc_mean*100, old_test_acc_std*100, old_val_acc_mean*100, old_val_acc_std*100, test_acc_mean*100, test_acc_std*100, val_acc_mean*100, val_acc_std*100))
        else:
            print ("Update configs of (%s, %s) failed:   %5.2f±%4.2f / %5.2f±%4.2f -> %5.2f±%4.2f / %5.2f±%4.2f " % (model, dataset, old_test_acc_mean*100, old_test_acc_std*100, old_val_acc_mean*100, old_val_acc_std*100, test_acc_mean*100, test_acc_std*100, val_acc_mean*100, val_acc_std*100)) 

def experiment(dataset, model=None, trial_log_path=None, update_configs=False, **kwargs):
    if model is None:
        model = "autognn"
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
    if isinstance(model[0], nn.Module):
        args.model = [x.model_name for x in model]
    print(args)
    args.dataset = dataset
    args.model = model

    if args.max_epoch is not None:
        warnings.warn("The max_epoch is deprecated and will be removed in the future, please use epochs instead!")
        args.epochs = args.max_epoch

    if len(model) == 1 and isinstance(model[0], str) and model[0] == "autognn":
        if not hasattr(args, "search_space"):
            args.search_space = default_search_space
        if not hasattr(args, "seed"):
            args.seed = [1, 2]
        if not hasattr(args, "n_trials"):
            args.n_trials = 20

    if hasattr(args, "search_space"):
        trial_result_dict, trial_log_dict = auto_experiment(args)
        
        if trial_log_path:
            with open(trial_log_path, 'w') as file:
                json.dump(trial_log_dict, file, indent=4, ensure_ascii=False)
        
        if update_configs:
            result_update(trial_log_dict, args.devices)

        return (trial_result_dict, trial_log_dict)

    return raw_experiment(args)
