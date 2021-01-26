import torch
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "hidden_size": 16,
        "eval_type": "all",
        "cpu": not cuda_available,
        "checkpoint": False,
        "device_id": [0],
    }
    return build_args_from_dict(default_dict)


def test_gatne_amazon():
    args = get_default_args()
    args.task = "multiplex_link_prediction"
    args.dataset = "amazon"
    args.model = "gatne"
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.schema = None
    args.epoch = 1
    args.batch_size = 128
    args.edge_dim = 5
    args.att_dim = 5
    args.negative_samples = 5
    args.neighbor_samples = 5
    task = build_task(args)
    ret = task.train()
    assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


def test_gatne_twitter():
    args = get_default_args()
    args.task = "multiplex_link_prediction"
    args.dataset = "twitter"
    args.model = "gatne"
    args.eval_type = ["1"]
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.schema = None
    args.epoch = 1
    args.batch_size = 128
    args.edge_dim = 5
    args.att_dim = 5
    args.negative_samples = 5
    args.neighbor_samples = 5
    task = build_task(args)
    ret = task.train()
    assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


def test_prone_amazon():
    args = get_default_args()
    args.task = "multiplex_link_prediction"
    args.dataset = "amazon"
    args.model = "prone"
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


def test_prone_youtube():
    args = get_default_args()
    args.task = "multiplex_link_prediction"
    args.dataset = "youtube"
    args.model = "prone"
    args.eval_type = ["1"]
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    task = build_task(args)
    ret = task.train()
    assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


if __name__ == "__main__":
    test_gatne_amazon()
    test_gatne_twitter()
    test_prone_amazon()
    test_prone_youtube()
