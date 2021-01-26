import torch
import random
import numpy as np

from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def get_strategies_for_pretrain_args():
    cuda_available = torch.cuda.is_available()
    args = {
        "dataset": "test_bio",
        "model": "stpgnn",
        "task": "pretrain",
        "batch_size": 32,
        "num_layers": 2,
        "JK": "last",
        "hidden_size": 32,
        "num_workers": 2,
        "finetune": False,
        "dropout": 0.5,
        "lr": 0.001,
        "cpu": not cuda_available,
        "device_id": [0],
        "weight_decay": 5e-4,
        "max_epoch": 3,
        "patience": 2,
        "output_model_file": "./saved",
        "l1": 1,
        "l2": 2,
        "checkpoint": False,
    }
    return build_args_from_dict(args)


def test_stpgnn_infomax():
    args = get_strategies_for_pretrain_args()
    args.pretrain_task = "infomax"
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_stpgnn_contextpred():
    args = get_strategies_for_pretrain_args()
    args.negative_samples = 1
    args.center = 0
    args.l1 = 1.0
    args.pretrain_task = "context"
    for mode in ["cbow", "skipgram"]:
        args.mode = mode
        task = build_task(args)
        ret = task.train()
        assert 0 < ret["Acc"] <= 1


def test_stpgnn_mask():
    args = get_strategies_for_pretrain_args()
    args.pretrain_task = "mask"
    args.mask_rate = 0.15
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_stpgnn_supervised():
    args = get_strategies_for_pretrain_args()
    args.pretrain_task = "supervised"
    args.pooling = "mean"
    args.load_path = None
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_stpgnn_finetune():
    args = get_strategies_for_pretrain_args()
    args.pretrain_task = "infomax"
    args.pooling = "mean"
    args.finetune = True
    args.load_path = "./saved/context.pth"
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_chem_infomax():
    args = get_strategies_for_pretrain_args()
    args.dataset = "test_chem"
    args.pretrain_task = "infomax"
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_chem_contextpred():
    args = get_strategies_for_pretrain_args()
    args.dataset = "test_chem"
    args.negative_samples = 1
    args.center = 0
    args.l1 = 1.0
    args.pretrain_task = "context"
    for mode in ["cbow", "skipgram"]:
        args.mode = mode
        task = build_task(args)
        ret = task.train()
        assert 0 < ret["Acc"] <= 1


def test_chem_mask():
    args = get_strategies_for_pretrain_args()
    args.dataset = "test_chem"
    args.pretrain_task = "mask"
    args.mask_rate = 0.15
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_chem_supervised():
    args = get_strategies_for_pretrain_args()
    args.dataset = "test_chem"
    args.pretrain_task = "supervised"
    args.pooling = "mean"
    args.load_path = None
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_bbbp():
    args = get_strategies_for_pretrain_args()
    args.dataset = "bbbp"
    args.pretrain_task = "infomax"
    args.pooling = "mean"
    args.finetune = True
    args.load_path = "./saved/context.pth"
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


def test_bace():
    args = get_strategies_for_pretrain_args()
    args.dataset = "bace"
    args.pretrain_task = "infomax"
    args.pooling = "mean"
    args.finetune = True
    args.load_path = "./saved/context.pth"
    task = build_task(args)
    ret = task.train()
    assert 0 < ret["Acc"] <= 1


if __name__ == "__main__":
    # test_stpgnn_infomax()
    # test_stpgnn_contextpred()
    # test_stpgnn_mask()
    # test_stpgnn_supervised()
    # test_stpgnn_finetune()
    # test_chem_contextpred()
    # test_chem_infomax()
    # test_chem_mask()
    test_chem_supervised()
    test_bace()
    test_bbbp()
