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
    }
    return build_args_from_dict(default_dict)


def test_deepwalk_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "deepwalk"
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_line_wikipedia():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "wikipedia"
    args.model = "line"
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 20
    args.alpha = 0.025
    args.order = 1
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_node2vec_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "node2vec"
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.p = 1.0
    args.q = 1.0
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_hope_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "hope"
    args.beta = 0.001
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_grarep_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "grarep"
    args.step = 1
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_netmf_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "netmf"
    args.window_size = 2
    args.rank = 32
    args.negative = 3
    args.is_large = False
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_netsmf_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "netsmf"
    args.window_size = 3
    args.negative = 1
    args.num_round = 2
    args.worker = 5
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_prone_flickr():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "flickr"
    args.model = "prone"
    args.step = 3
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_sdne_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "sdne"
    args.hidden_size1 = 100
    args.hidden_size2 = 16
    args.droput = 0.2
    args.alpha = 0.01
    args.beta = 5
    args.nu1 = 1e-4
    args.nu2 = 1e-3
    args.max_epoch = 1
    args.lr = 0.001
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["ROC_AUC"] <= 1


def test_dngr_ppi():
    args = get_default_args()
    args.task = "link_prediction"
    args.dataset = "ppi"
    args.model = "dngr"
    args.hidden_size1 = 100
    args.hidden_size2 = 16
    args.noise = 0.2
    args.alpha = 0.01
    args.step = 2
    args.max_epoch = 1
    args.lr = 0.001
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
    # test_deepwalk_ppi()
    # test_line_wikipedia()
    # test_node2vec_ppi()
    # test_hope_ppi()
    # test_grarep_ppi()
    # test_netmf_ppi()
    # test_netsmf_ppi()
    # test_prone_flickr()
    # test_sdne_ppi()
    # test_dngr_ppi()
    #
    # test_rgcn_wn18()
    # test_compgcn_wn18rr()
    #
    # test_distmult_fb13s()
    # test_rotate_fb13s()
    # test_transe_fb13s()
    # test_complex_fb13s()
    test_gcn_cora()
