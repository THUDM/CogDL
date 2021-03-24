import torch

from torch import argsort
from cogdl import options
from cogdl.tasks import build_task, register_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict


def get_default_args():
    default_dict = {
        "hidden_size": 16,
        "negative_ratio": 3,
        "patience": 1,
        "max_epoch": 1,
        "cpu": True,
        "checkpoint": False,
        "save_dir": ".",
        "device_id": [0],
    }
    return build_args_from_dict(default_dict)


def test_prone_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi-ne"
    args.model = "prone"
    args.step = 3
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def get_kg_default_args():
    default_dict = {
        "max_epoch": 2,
        "num_bases": 5,
        "num_layers": 2,
        "hidden_size": 40,
        "penalty": 0.001,
        "sampling_rate": 0.001,
        "dropout": 0.3,
        "evaluate_interval": 2,
        "patience": 20,
        "lr": 0.001,
        "weight_decay": 0,
        "negative_ratio": 3,
        "cpu": True,
        "checkpoint": False,
        "save_dir": ".",
        "device_id": [0],
    }
    return build_args_from_dict(default_dict)


def get_nums(dataset, args):
    data = dataset[0]
    args.num_entities = len(torch.unique(data.edge_index))
    args.num_rels = len(torch.unique(data.edge_attr))
    return args


def test_rgcn_wn18():
    args = get_kg_default_args()
    args.self_dropout = 0.2
    args.self_loop = True
    args.dataset = "wn18"
    args.model = "rgcn"
    args.task = "link_prediction"
    args.regularizer = "basis"
    dataset = build_dataset(args)
    args = get_nums(dataset, args)
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["MRR"] <= 1


def test_compgcn_wn18rr():
    args = get_kg_default_args()
    args.lbl_smooth = 0.1
    args.score_func = "distmult"
    args.dataset = "wn18rr"
    args.model = "compgcn"
    args.task = "link_prediction"
    args.regularizer = "basis"
    dataset = build_dataset(args)
    args = get_nums(dataset, args)
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["MRR"] <= 1


def get_kge_default_args():
    default_dict = {
        "embedding_size": 8,
        "nentity": None,
        "nrelation": None,
        "do_train": True,
        "do_valid": False,
        "save_path": ".",
        "init_checkpoint": None,
        "save_checkpoint_steps": 100,
        "double_entity_embedding": False,
        "double_relation_embedding": False,
        "negative_adversarial_sampling": False,
        "negative_sample_size": 1,
        "batch_size": 64,
        "test_batch_size": 100,
        "uni_weight": False,
        "learning_rate": 0.0001,
        "warm_up_steps": None,
        "max_epoch": 10,
        "log_steps": 100,
        "test_log_steps": 100,
        "gamma": 12,
        "regularization": 0.0,
        "cuda": False,
        "cpu": True,
        "checkpoint": False,
        "save_dir": ".",
        "device_id": [0],
    }
    return build_args_from_dict(default_dict)


def test_distmult_fb13s():
    args = get_kge_default_args()
    args.dataset = "fb13s"
    args.model = "distmult"
    args.task = "link_prediction"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["MRR"] <= 1


def test_rotate_fb13s():
    args = get_kge_default_args()
    args.dataset = "fb13s"
    args.model = "rotate"
    args.task = "link_prediction"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["MRR"] <= 1


def test_transe_fb13s():
    args = get_kge_default_args()
    args.dataset = "fb13s"
    args.model = "transe"
    args.task = "link_prediction"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["MRR"] <= 1


def test_complex_fb13s():
    args = get_kge_default_args()
    args.dataset = "fb13s"
    args.model = "complex"
    args.task = "link_prediction"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["MRR"] <= 1


def get_gnn_link_prediction_args():
    args = {
        "hidden_size": 32,
        "dataset": "cora",
        "model": "gcn",
        "task": "link_prediction",
        "lr": 0.005,
        "weight_decay": 5e-4,
        "max_epoch": 60,
        "patience": 2,
        "num_layers": 2,
        "evaluate_interval": 1,
        "cpu": True,
        "device_id": [0],
        "dropout": 0.5,
        "checkpoint": False,
        "save_dir": ".",
    }
    return build_args_from_dict(args)


def test_gcn_cora():
    args = get_gnn_link_prediction_args()
    print(args.evaluate_interval)
    task = build_task(args)
    ret = task.train()
    assert 0.5 <= ret["AUC"] <= 1.0


if __name__ == "__main__":
    test_prone_ppi()

    test_rgcn_wn18()
    test_compgcn_wn18rr()

    test_distmult_fb13s()
    test_rotate_fb13s()
    test_transe_fb13s()
    test_complex_fb13s()

    test_gcn_cora()
