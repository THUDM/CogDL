from cogdl.options import get_default_args
from cogdl.experiments import train


default_dict = {
    "hidden_size": 8,
    "dropout": 0.5,
    "patience": 1,
    "epochs": 1,
    "device_id": [0],
    "cpu": True,
    "lr": 0.001,
    "weight_decay": 5e-4,
    "checkpoint": False,
    "seed": 0,
}


def get_default_args_hgnn(dataset, model, dw="heterogeneous_gnn_dw", mw="heterogeneous_gnn_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict.items():
        args.__setattr__(key, value)
    return args


def test_gtn_gtn_imdb():
    args = get_default_args_hgnn(dataset="gtn-imdb", model="gtn")
    args.num_channels = 2
    args.num_layers = 2
    ret = train(args)
    assert ret["test_acc"] >= 0 and ret["test_acc"] <= 1


def test_han_gtn_acm():
    args = get_default_args_hgnn(dataset="gtn-acm", model="han")
    args.num_layers = 2
    ret = train(args)
    assert ret["test_acc"] >= 0 and ret["test_acc"] <= 1


def test_han_gtn_dblp():
    args = get_default_args_hgnn(dataset="gtn-dblp", model="han")
    args.num_layers = 2
    ret = train(args)
    assert ret["test_acc"] >= 0 and ret["test_acc"] <= 1


def test_han_han_imdb():
    args = get_default_args_hgnn(dataset="han-imdb", model="han")
    args.num_layers = 2
    ret = train(args)
    assert ret["test_acc"] >= 0 and ret["test_acc"] <= 1


def test_han_han_acm():
    args = get_default_args_hgnn(dataset="han-acm", model="han")
    args.num_layers = 2
    ret = train(args)
    assert ret["test_acc"] >= 0 and ret["test_acc"] <= 1


default_dict_emb = {
    "hidden_size": 16,
    "cpu": True,
    "enhance": False,
    "save_dir": "./embedding",
    "checkpoint": False,
    "device_id": [0],
}


def get_default_args_emb(dataset, model, dw="heterogeneous_embedding_dw", mw="heterogeneous_embedding_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict_emb.items():
        args.__setattr__(key, value)
    return args


def test_metapath2vec_gtn_acm():
    args = get_default_args_emb(dataset="gtn-acm", model="metapath2vec")
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.schema = "No"
    ret = train(args)
    assert ret["f1"] > 0


def test_metapath2vec_gtn_imdb():
    args = get_default_args_emb(dataset="gtn-imdb", model="metapath2vec")
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.schema = "No"
    ret = train(args)
    assert ret["f1"] > 0


def test_pte_gtn_imdb():
    args = get_default_args_emb(dataset="gtn-imdb", model="pte")
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 10
    args.alpha = 0.025
    args.order = "No"
    ret = train(args)
    assert ret["f1"] > 0


def test_pte_gtn_dblp():
    args = get_default_args_emb(dataset="gtn-dblp", model="pte")
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 10
    args.alpha = 0.025
    args.order = "No"
    ret = train(args)
    assert ret["f1"] > 0


def test_hin2vec_dblp():
    args = get_default_args_emb(dataset="gtn-dblp", model="hin2vec")
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 1000
    args.hop = 2
    args.epochs = 1
    args.lr = 0.025
    args.cpu = True
    ret = train(args)
    assert ret["f1"] > 0


if __name__ == "__main__":
    test_gtn_gtn_imdb()
    test_han_gtn_acm()
    test_han_gtn_dblp()
    test_han_han_imdb()
    test_han_han_acm()

    test_metapath2vec_gtn_acm()
    test_metapath2vec_gtn_imdb()
    test_pte_gtn_imdb()
    test_pte_gtn_dblp()
    test_hin2vec_dblp()
