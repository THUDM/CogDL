import torch

from torch import argsort
from cogdl import options
from cogdl.tasks import build_task, register_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    default_dict = {'hidden_size': 16,
                    'negative_ratio': 3,
                    'patience': 1,
                    'max_epoch': 1,
                    'cpu': True}
    return build_args_from_dict(default_dict)

def test_deepwalk_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'deepwalk'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1

def test_line_wikipedia():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'wikipedia'
    args.model = 'line'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 20
    args.alpha = 0.025
    args.order = 1
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1

def test_node2vec_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'node2vec'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.p = 1.0
    args.q = 1.0
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1

def test_hope_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'hope'
    dataset = build_dataset(args)
    args.beta = 0.001
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1


def test_grarep_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'grarep'
    dataset = build_dataset(args)
    args.step = 1
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1
    
def test_netmf_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'netmf'
    dataset = build_dataset(args)
    args.window_size = 2
    args.rank = 32
    args.negative = 3
    args.is_large = False
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1

def test_netsmf_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'netsmf'
    dataset = build_dataset(args)
    args.window_size = 3
    args.negative = 1
    args.num_round = 2
    args.worker = 5
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1

def test_prone_flickr():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'flickr'
    args.model = 'prone'
    dataset = build_dataset(args)
    args.step = 3
    args.theta = 0.5
    args.mu = 0.2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1

def test_sdne_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'sdne'
    dataset = build_dataset(args)
    args.hidden_size1 = 100
    args.hidden_size2 = 16
    args.droput = 0.2
    args.alpha = 0.01
    args.beta = 5
    args.nu1 = 1e-4
    args.nu2 = 1e-3
    args.max_epoch = 1
    args.lr = 0.001
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1

def test_dngr_ppi():
    args = get_default_args()
    args.task = 'link_prediction'
    args.dataset = 'ppi'
    args.model = 'dngr'
    dataset = build_dataset(args)
    args.hidden_size1 = 100
    args.hidden_size2 = 16
    args.noise = 0.2
    args.alpha = 0.01
    args.step = 2
    args.max_epoch = 1
    args.lr = 0.001
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['ROC_AUC'] >= 0 and ret['ROC_AUC'] <= 1


def get_kg_default_arfgs():
    default_dict = {
        "max_epoch": 2,
        "num_bases": 5,
        "num_layers": 2,
        "hidden_size": 200,
        "penalty": 0.1,
        "sampling_rate": 0.001,
        "dropout": 0.3,
        "evaluate_interval": 2,
        "patience": 20,
        "lr": 0.001,
        "weight_decay": 0,
        "negative_ratio": 3,
        "cpu": True,
    }
    return build_args_from_dict(default_dict)


def get_nums(dataset, args):
    data = dataset[0]
    args.num_entities = len(torch.unique(data.edge_index))
    args.num_rels = len(torch.unique(data.edge_attr))
    return args

def test_rgcn_wn18():
    args = get_kg_default_arfgs()
    args.self_dropout = 0.2
    args.self_loop = True
    args.dataset = "wn18"
    args.model = "rgcn"
    args.task = "link_prediction"
    args.regularizer = "basis"
    dataset = build_dataset(args)
    args = get_nums(dataset, args)
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret["MRR"] >= 0 and ret["MRR"] < 1


def test_rgcn_fb15k237():
    args = get_kg_default_arfgs()
    args.self_dropout = 0.2
    args.sampling_rate = 0.001
    args.self_loop = True
    args.dataset = "fb15k237"
    args.model = "rgcn"
    args.task = "link_prediction"
    args.regularizer = "basis"
    dataset = build_dataset(args)
    args = get_nums(dataset, args)
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret["MRR"] >= 0 and ret["MRR"] < 1


def test_compgcn_fb13():
    args = get_kg_default_arfgs()
    args.self_dropout = 0.2
    args.self_loop = True
    args.dataset = "fb13"
    args.model = "rgcn"
    args.task = "link_prediction"
    args.regularizer = "basis"
    dataset = build_dataset(args)
    args = get_nums(dataset, args)
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret["MRR"] >= 0 and ret["MRR"] < 1

def test_compgcn_wn18rr():
    args = get_kg_default_arfgs()
    args.lbl_smooth = 0.1
    args.score_func = "distmult"
    args.dataset = "wn18rr"
    args.model = "compgcn"
    args.task = "link_prediction"
    args.regularizer = "basis"
    dataset = build_dataset(args)
    args = get_nums(dataset, args)
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret["MRR"] >= 0 and ret["MRR"] < 1


if __name__ == "__main__":
    test_deepwalk_ppi()
    test_line_wikipedia()
    test_node2vec_ppi()
    test_hope_ppi()
    test_grarep_ppi()
    test_netmf_ppi()
    test_netsmf_ppi()
    test_prone_flickr()
    test_sdne_ppi()
    test_dngr_ppi()

    test_rgcn_wn18()
    test_rgcn_fb15k237()
    test_compgcn_fb13()
    test_compgcn_wn18rr()