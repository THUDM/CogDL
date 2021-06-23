from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def get_default_args():
    default_dict = {
        "task": "recommendation",
        "patience": 2,
        "device_id": [0],
        "max_epoch": 1,
        "cpu": True,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "evaluate_interval": 5,
        "num_workers": 4,
        "batch_size": 20480,
    }
    return build_args_from_dict(default_dict)


def test_lightgcn_ali():
    args = get_default_args()
    args.dataset = "ali"
    args.model = "lightgcn"
    args.Ks = [1]
    args.dim = 8
    args.l2 = 1e-4
    args.mess_dropout = False
    args.mess_dropout_rate = 0.0
    args.edge_dropout = False
    args.edge_dropout_rate = 0.0
    args.ns = "rns"
    args.K = 1
    args.n_negs = 1
    args.pool = "mean"
    args.context_hops = 1
    task = build_task(args)
    ret = task.train(unittest=True)
    assert ret["Recall"] >= 0


if __name__ == "__main__":
    test_lightgcn_ali()
