from sklearn.metrics import f1_score
import torch
from cogdl.utils import build_args_from_dict
from cogdl.utils import accuracy_evaluator, multilabel_evaluator, multiclass_evaluator


def test_build_args_from_dict():
    dic = {"arg1": "value1", "arg2": 2, "arg3": 0.3}
    args = build_args_from_dict(dic)

    assert args.arg1 == "value1"
    assert args.arg2 == 2
    assert args.arg3 == 0.3


def test_evaluator():
    pred = torch.randn(20, 5)
    target_one = torch.randint(0, 5, (20,))
    target_mult = torch.randint(0, 2, (20, 5)).float()

    loss_func, accuracy = accuracy_evaluator()
    _pred = torch.nn.functional.log_softmax(pred, dim=-1)
    _ = loss_func(_pred, target_one)
    acc = _pred.max(1)[1].eq(target_one).double().sum() / len(_pred)
    assert acc == accuracy(target_one, _pred)
    loss_func, multicls = multiclass_evaluator()
    _ = loss_func(_pred, target_one)
    f1 = f1_score(target_one, _pred.max(1)[1], average="micro")
    assert f1 == multicls(target_one, _pred)
    loss_func, multilabel = multilabel_evaluator()
    _ = loss_func(pred, target_mult)
    _pred = torch.zeros_like(pred)
    _pred[pred > 0] = 1
    f1 = f1_score(target_mult, _pred, average="micro")
    assert f1 == multilabel(target_mult, pred)


if __name__ == "__main__":
    test_build_args_from_dict()
