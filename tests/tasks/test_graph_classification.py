import torch
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "task": "graph_classification",
        "hidden_size": 32,
        "dropout": 0.5,
        "patience": 1,
        "max_epoch": 2,
        "cpu": not cuda_available,
        "lr": 0.001,
        "kfold": False,
        "seed": [0],
        "weight_decay": 5e-4,
        "gamma": 0.5,
        "train_ratio": 0.7,
        "test_ratio": 0.1,
        "device_id": [0 if cuda_available else "cpu"],
        "sampler": "none",
        "degree_feature": False,
        "checkpoint": False,
    }
    return build_args_from_dict(default_dict)


def add_diffpool_args(args):
    args.num_layers = 2
    args.num_pooling_layers = 1
    args.no_link_pred = False
    args.pooling_ratio = 0.15
    args.embedding_dim = 64
    args.hidden_size = 64
    args.batch_size = 20
    args.dropout = 0.1
    return args


def add_gin_args(args):
    args.epsilon = 0.0
    args.hidden_size = 32
    args.num_layers = 5
    args.num_mlp_layers = 2
    args.train_epsilon = True
    args.pooling = "sum"
    args.batch_size = 128
    return args


def add_dgcnn_args(args):
    args.hidden_size = 64
    args.batch_size = 20
    return args


def add_sortpool_args(args):
    args.hidden_size = 64
    args.batch_size = 20
    args.num_layers = 2
    args.out_channels = 32
    args.k = 30
    args.kernel_size = 5
    return args


def add_patchy_san_args(args):
    args.hidden_size = 64
    args.batch_size = 20
    args.sample = 10
    args.stride = 1
    args.neighbor = 10
    args.iteration = 2
    args.train_ratio = 0.7
    args.test_ratio = 0.1
    return args


def add_hgpsl_args(args):
    args.hidden_size = 128
    args.dropout = 0.0
    args.pooling = 0.5
    args.batch_size = 64
    args.train_ratio = 0.8
    args.test_ratio = 0.1
    args.lr = 0.001
    return args


def add_sagpool_args(args):
    args.hidden_size = 128
    args.batch_size = 20
    args.train_ratio = 0.7
    args.test_ratio = 0.1
    args.pooling_ratio = 0.5
    args.pooling_layer_type = "gcnconv"
    return args


def test_gin_mutag():
    args = get_default_args()
    args = add_gin_args(args)
    args.dataset = "mutag"
    args.model = "gin"
    args.batch_size = 20
    for kfold in [True, False]:
        args.kfold = kfold
        args.seed = 0
        task = build_task(args)
        ret = task.train()
        assert ret["Acc"] > 0


def test_gin_imdb_binary():
    args = get_default_args()
    args = add_gin_args(args)
    args.dataset = "imdb-b"
    args.model = "gin"
    args.degree_feature = True
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_gin_proteins():
    args = get_default_args()
    args = add_gin_args(args)
    args.dataset = "proteins"
    args.model = "gin"
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_diffpool_mutag():
    args = get_default_args()
    args = add_diffpool_args(args)
    args.dataset = "mutag"
    args.model = "diffpool"
    args.batch_size = 5
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_diffpool_proteins():
    args = get_default_args()
    args = add_diffpool_args(args)
    args.dataset = "proteins"
    args.model = "diffpool"
    args.batch_size = 20
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_dgcnn_proteins():
    args = get_default_args()
    args = add_dgcnn_args(args)
    args.dataset = "proteins"
    args.model = "dgcnn"
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_dgcnn_imdb_binary():
    args = get_default_args()
    args = add_dgcnn_args(args)
    args.dataset = "imdb-b"
    args.model = "dgcnn"
    args.degree_feature = True
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_sortpool_mutag():
    args = get_default_args()
    args = add_sortpool_args(args)
    args.dataset = "mutag"
    args.model = "sortpool"
    args.batch_size = 20
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_sortpool_proteins():
    args = get_default_args()
    args = add_sortpool_args(args)
    args.dataset = "proteins"
    args.model = "sortpool"
    args.batch_size = 20
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_patchy_san_mutag():
    args = get_default_args()
    args = add_patchy_san_args(args)
    args.dataset = "mutag"
    args.model = "patchy_san"
    args.batch_size = 20
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_patchy_san_proteins():
    args = get_default_args()
    args = add_patchy_san_args(args)
    args.dataset = "proteins"
    args.model = "patchy_san"
    args.batch_size = 20
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_hgpsl_proteins():
    args = get_default_args()
    args = add_hgpsl_args(args)
    args.dataset = "proteins"
    args.model = "hgpsl"
    args.sample_neighbor = (True,)
    args.sparse_attention = (True,)
    args.structure_learning = (True,)
    args.lamb = 1.0
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


def test_sagpool_mutag():
    args = get_default_args()
    args = add_sagpool_args(args)
    args.dataset = "mutag"
    args.model = "sagpool"
    args.batch_size = 20
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] >= 0


def test_sagpool_proteins():
    args = get_default_args()
    args = add_sagpool_args(args)
    args.dataset = "proteins"
    args.model = "sagpool"
    args.batch_size = 20
    task = build_task(args)
    ret = task.train()
    assert ret["Acc"] > 0


if __name__ == "__main__":

    test_gin_imdb_binary()
    test_gin_mutag()
    test_gin_proteins()

    test_sortpool_mutag()
    test_sortpool_proteins()

    test_diffpool_mutag()
    test_diffpool_proteins()

    test_dgcnn_proteins()
    test_dgcnn_imdb_binary()

    test_patchy_san_mutag()
    test_patchy_san_proteins()

    test_hgpsl_proteins()

    test_sagpool_mutag()
    test_sagpool_proteins()
