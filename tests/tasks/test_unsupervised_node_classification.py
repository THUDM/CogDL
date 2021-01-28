import numpy as np
import torch

from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict


def get_default_args():
    default_dict = {
        "hidden_size": 16,
        "num_shuffle": 1,
        "cpu": True,
        "enhance": None,
        "save_dir": "./embedding",
        "task": "unsupervised_node_classification",
        "checkpoint": False,
        "load_emb_path": None,
        "training_percents": [0.1],
    }
    return build_args_from_dict(default_dict)


def test_deepwalk_wikipedia():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "wikipedia"
    args.model = "deepwalk"
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_line_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "line"
    args.walk_length = 1
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 20
    args.alpha = 0.025
    args.order = 1
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_node2vec_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
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
    assert ret["Micro-F1 0.1"] > 0


def test_hope_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "hope"
    args.beta = 0.001
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_prone_module():
    from cogdl.layers.prone_module import propagate
    import scipy.sparse as sp

    data = np.ones(400)
    edge_index = np.random.randint(0, 100, (2, 200))
    row = np.concatenate((edge_index[0], edge_index[1]))
    col = np.concatenate((edge_index[1], edge_index[0]))

    print(row.shape, col.shape)
    matrix = sp.csr_matrix((data, (row, col)), shape=(100, 100))
    emb = np.random.randn(100, 20)
    for module in ["heat", "ppr", "gaussian", "prone", "sc"]:
        res = propagate(matrix, emb, module)
        assert res.shape == emb.shape


def test_grarep_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "grarep"
    args.step = 1
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_netmf_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "netmf"
    args.window_size = 2
    args.rank = 32
    args.negative = 3
    args.is_large = False
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_netsmf_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "netsmf"
    args.window_size = 3
    args.negative = 1
    args.num_round = 2
    args.worker = 5
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_prone_blogcatalog():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "blogcatalog"
    args.model = "prone"
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_prone_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "prone"
    args.enhance = "prone++"
    args.max_evals = 3
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_prone_usa_airport():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "usa-airport"
    args.model = "prone"
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_spectral_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "spectral"
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def test_sdne_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
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
    assert ret["Micro-F1 0.1"] > 0


def test_dngr_ppi():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "ppi-ne"
    args.model = "dngr"
    args.hidden_size1 = 100
    args.hidden_size2 = 16
    args.noise = 0.2
    args.alpha = 0.01
    args.step = 3
    args.max_epoch = 1
    args.lr = 0.001
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


def get_unsupervised_nn_args():
    default_dict = {
        "hidden_size": 16,
        "num_layers": 2,
        "lr": 0.01,
        "dropout": 0.0,
        "patience": 1,
        "max_epoch": 1,
        "cpu": not torch.cuda.is_available(),
        "weight_decay": 5e-4,
        "num_shuffle": 2,
        "save_dir": "./embedding",
        "enhance": None,
        "device_id": [
            0,
        ],
        "task": "unsupervised_node_classification",
        "checkpoint": False,
        "load_emb_path": None,
        "training_percents": [0.1],
    }
    return build_args_from_dict(default_dict)


def build_nn_dataset(args):
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    return args, dataset


def test_unsupervised_graphsage():
    args = get_unsupervised_nn_args()
    args.negative_samples = 10
    args.walk_length = 5
    args.sample_size = [5, 5]
    args.patience = 20
    args.task = "unsupervised_node_classification"
    args.dataset = "cora"
    args.max_epochs = 2
    args.model = "unsup_graphsage"
    args, dataset = build_nn_dataset(args)
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0

    args.checkpoint = True
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0

    args.load_emb_path = args.save_dir + "/" + args.model + "_" + args.dataset + ".npy"
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_dgi():
    args = get_unsupervised_nn_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "cora"
    args.activation = "relu"
    args.sparse = True
    args.max_epochs = 2
    args.model = "dgi"
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_mvgrl():
    args = get_unsupervised_nn_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "cora"
    args.max_epochs = 2
    args.model = "mvgrl"
    args.sparse = False
    args.sample_size = 2000
    args.batch_size = 4
    args, dataset = build_nn_dataset(args)
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_grace():
    args = get_unsupervised_nn_args()
    args.model = "grace"
    args.num_layers = 2
    args.max_epoch = 2
    args.drop_feature_rates = [0.1, 0.2]
    args.drop_edge_rates = [0.2, 0.3]
    args.activation = "relu"
    args.proj_hidden_size = 32
    args.tau = 0.5
    args.dataset = "cora"
    args, dataset = build_nn_dataset(args)

    for bs in [-1, 512]:
        args.batch_size = bs
        task = build_task(args)
        ret = task.train()
        assert ret["Acc"] > 0


def test_gcc_usa_airport():
    args = get_default_args()
    args.task = "unsupervised_node_classification"
    args.dataset = "usa-airport"
    args.model = "gcc"
    args.load_path = "./saved/gcc_pretrained.pth"
    task = build_task(args)
    ret = task.train()
    assert ret["Micro-F1 0.1"] > 0


if __name__ == "__main__":
    test_unsupervised_graphsage()
    test_dgi()
    test_mvgrl()
    test_deepwalk_wikipedia()
    test_line_ppi()
    test_node2vec_ppi()
    test_hope_ppi()
    test_grarep_ppi()
    test_netmf_ppi()
    test_netsmf_ppi()
    test_prone_blogcatalog()
    test_sdne_ppi()
    test_dngr_ppi()
    test_prone_ppi()
    test_prone_usa_airport()
    test_spectral_ppi()
    test_gcc_usa_airport()
    test_grace()
    test_prone_module()
