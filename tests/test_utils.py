from sklearn.metrics import f1_score
import torch
from cogdl.utils import build_args_from_dict
from cogdl.utils import accuracy, multiclass_f1, multilabel_f1, bce_with_logits_loss, cross_entropy_loss


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

    def f(x):
        return round(float(x), 5)

    _ = cross_entropy_loss(pred, target_one)
    _pred = torch.nn.functional.log_softmax(pred, dim=-1)
    acc = _pred.max(1)[1].eq(target_one).double().sum() / len(_pred)
    assert f(acc) == f(accuracy(_pred, target_one))
    f1 = f1_score(target_one, _pred.max(1)[1], average="micro")
    assert f(f1) == f(multiclass_f1(_pred, target_one))
    _ = bce_with_logits_loss(pred, target_mult)
    _pred = torch.zeros_like(pred)
    _pred[pred > 0] = 1
    f1 = f1_score(target_mult, _pred, average="micro")
    assert f(f1) == f(multilabel_f1(pred, target_mult))


if __name__ == "__main__":
    test_build_args_from_dict()
