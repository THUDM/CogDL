import errno
import itertools
import os
import os.path as osp
import random
import shutil
from collections import defaultdict
from urllib import request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate

from .graph_utils import coo2csr_index


class ArgClass(object):
    def __init__(self):
        pass


def build_args_from_dict(dic):
    args = ArgClass()
    for key, value in dic.items():
        args.__setattr__(key, value)
    return args


def update_args_from_dict(args, dic):
    for key, value in dic.items():
        args.__setattr__(key, value)
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def untar(path, fname, deleteTar=True):
    """
    Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print("unpacking " + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, name=None, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        name (string): saved filename.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    if log:
        print("Downloading", url)

    makedirs(folder)

    try:
        data = request.urlopen(url)
    except Exception as e:
        print(e)
        print("Failed to download the dataset.")
        print(f"Please download the dataset manually and put it under {folder}.")
        exit(1)

    if name is None:
        filename = url.rpartition("/")[2]
    else:
        filename = name
    path = osp.join(folder, filename)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def identity_act(input):
    return input


def get_activation(act: str, inplace=False):
    if act == "relu":
        return nn.ReLU(inplace=inplace)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "gelu":
        return nn.GELU()
    elif act == "prelu":
        return nn.PReLU()
    elif act == "identity":
        return identity_act
    else:
        return identity_act


def get_norm_layer(norm: str, channels: int):
    """
    Args:
        norm: str
            type of normalization: `layernorm`, `batchnorm`, `instancenorm`
        channels: int
            size of features for normalization
    """
    if norm == "layernorm":
        return torch.nn.LayerNorm(channels)
    elif norm == "batchnorm":
        return torch.nn.BatchNorm1d(channels)
    elif norm == "instancenorm":
        return torch.nn.InstanceNorm1d(channels)
    else:
        return torch.nn.Identity()


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def batch_sum_pooling(x, batch):
    batch_size = int(torch.max(batch.cpu())) + 1
    # batch_size = len(torch.unique(batch))
    res = torch.zeros(batch_size, x.size(1)).to(x.device)
    return res.scatter_add_(dim=0, index=batch.unsqueeze(-1).expand_as(x), src=x)


def batch_mean_pooling(x, batch):
    values, counts = torch.unique(batch, return_counts=True)
    res = torch.zeros(len(values), x.size(1)).to(x.device)
    res = res.scatter_add_(dim=0, index=batch.unsqueeze(-1).expand_as(x), src=x)
    return res / counts.unsqueeze(-1)


def batch_max_pooling(x, batch):
    if torch.cuda.is_available() and str(x.device) != "cpu":
        try:
            from cogdl.operators.scatter_max import scatter_max

            col = torch.arange(0, len(batch)).to(x.device)
            rowptr, colind = coo2csr_index(batch, col, num_nodes=batch.max().item() + 1)
            x = scatter_max(rowptr.int(), colind.int(), x)
            return x
        except Exception:
            pass

    from torch_scatter import scatter_max

    x, _ = scatter_max(x, batch, dim=0)
    return x


def tabulate_results(results_dict):
    # Average for different seeds
    # {"model1_dataset": [dict(acc=1), dict(acc=2)], "model2_dataset": [dict(acc=1),...]}
    tab_data = []
    for variant in results_dict:
        results = np.array([list(res.values()) for res in results_dict[variant]])
        if isinstance(variant[1], nn.Module):
            variant = (variant[0], variant[1].model_name)
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(np.mean(results, axis=0).tolist(), np.std(results, axis=0).tolist(),),
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
        results_dict[(model_name, datasets[i // num_seed])].append(res)
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=table_header, tablefmt="github"))


def split_dataset_general(dataset, args):
    droplast = args.model == "diffpool"

    train_size = int(len(dataset) * args.train_ratio)
    test_size = int(len(dataset) * args.test_ratio)
    index = list(range(len(dataset)))
    random.shuffle(index)

    train_index = index[:train_size]
    test_index = index[-test_size:]

    bs = args.batch_size
    train_dataset = dict(dataset=[dataset[i] for i in train_index], batch_size=bs, drop_last=droplast)
    test_dataset = dict(dataset=[dataset[i] for i in test_index], batch_size=bs, drop_last=droplast)
    if args.train_ratio + args.test_ratio < 1:
        val_index = index[train_size:-test_size]
        valid_dataset = dict(dataset=[dataset[i] for i in val_index], batch_size=bs, drop_last=droplast)
    else:
        valid_dataset = test_dataset
    return train_dataset, valid_dataset, test_dataset


def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def build_model_path(args, model_name):
    if not hasattr(args, "save_model_path"):
        args.save_model_path = ""
    if model_name == "gcc":
        if hasattr(args, "pretrain") and args.pretrain:
            model_name_path = "{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}".format(
                "Pretrain" if not args.finetune else "FT", 
                '_'.join([x.replace('gcc_', '').replace('_', '-') for x in args.dataset.split(' ')]),
                args.gnn_model,
                args.num_layers,
                args.lr,
                args.weight_decay,
                args.batch_size,
                args.hidden_size,
                args.num_samples,
                args.nce_t,
                args.nce_k,
                args.rw_hops,
                args.restart_prob,
                args.aug,
                args.finetune,
                args.degree_embedding_size,
                args.positional_embedding_size,
                args.momentum,
            )

            args.save_model_path = os.path.join(args.save_model_path, model_name_path)
            os.makedirs(args.save_model_path, exist_ok=True)
    return args


if __name__ == "__main__":
    args = build_args_from_dict({"a": 1, "b": 2})
    print(args.a, args.b)
