from cogdl.options import get_default_args
from cogdl.experiments import train



default_dict_kg = {
    "epochs": 2,
    "batch_size": 1024,
    "cpu":True,
    "lr": 0.001,
    "negative_ratio": 3,
    "checkpoint": False,
    "save_dir": ".",
    "device_id": [0],
    "actnn": False,
    "do_test":False,
    "do_valid":False ,
    "eval_step":3,
}


def get_default_args_kg(dataset, model, dw="triple_link_prediction_dw", mw="triple_link_prediction_mw"):
    args = get_default_args(dataset=dataset, model=model, dw=dw, mw=mw)
    for key, value in default_dict_kg.items():
        args.__setattr__(key, value)
    return args


def test_transe_fb15k():
    args = get_default_args_kg(dataset="fb15k", model="transe")
    ret = train(args)
    #assert 0 <= ret["mrr"] <= 1


def test_complex_fb15k():
    args = get_default_args_kg(dataset="fb15k", model="complex")
    args.double_entity_embedding = True
    args.double_relation_embedding=True
    ret = train(args)
    #assert 0 <= ret["mrr"] <= 1


def test_distmult_wn18():
    args = get_default_args_kg(dataset="wn18", model="distmult")
    ret = train(args)
    #assert 0 <= ret["mrr"] <= 1

def test_rotate_wn18():
    args = get_default_args_kg(dataset="wn18", model="rotate")
    args.double_entity_embedding = True
    ret = train(args)
    #assert 0 <= ret["mrr"] <= 1


if __name__ == "__main__":
    test_transe_fb15k()
    test_complex_fb15k()
    test_distmult_wn18()
    test_rotate_wn18()
    