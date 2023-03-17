import torch

from cogdl.options import get_default_args
from cogdl.experiments import train

cuda_available = torch.cuda.is_available()
default_dict = {
    "hidden_size": 16,
    "eval_type": "all",
    "cpu": not cuda_available,
    "checkpoint": False,
    "device_id": [0],
    "activation": "relu",
    "residual": False,
    "norm": None,
}


def get_default_args_multiplex(dataset, model, dw="multiplex_embedding_dw", mw="multiplex_embedding_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args


def test_gatne_amazon():
    args = get_default_args_multiplex(dataset="amazon", model="gatne")
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.schema = None
    args.epochs = 1
    args.batch_size = 128
    args.edge_dim = 5
    args.att_dim = 5
    args.negative_samples = 5
    args.neighbor_samples = 5
    ret = train(args)
    assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


def test_gatne_twitter():
    args = get_default_args_multiplex(dataset="twitter", model="gatne")
    args.eval_type = ["1"]
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.schema = None
    args.epochs = 1
    args.batch_size = 128
    args.edge_dim = 5
    args.att_dim = 5
    args.negative_samples = 5
    args.neighbor_samples = 5
    ret = train(args)
    assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


def test_prone_amazon():
    args = get_default_args_multiplex(dataset="amazon", model="prone")
    args.step = 5
    args.theta = 0.5
    args.mu = 0.2
    ret = train(args)
    assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


# def test_prone_youtube():
#     args = get_default_args_multiplex(dataset="youtube", model="prone")
#     args.eval_type = ["1"]
#     args.step = 5
#     args.theta = 0.5
#     args.mu = 0.2
#     ret = train(args)
#     assert ret["ROC_AUC"] >= 0 and ret["ROC_AUC"] <= 1


if __name__ == "__main__":
    test_gatne_amazon()
    test_gatne_twitter()
    test_prone_amazon()
    # test_prone_youtube()
