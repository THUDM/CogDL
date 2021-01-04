from cogdl.utils import build_args_from_dict


def test_build_args_from_dict():
    dic = {"arg1": "value1", "arg2": 2, "arg3": 0.3}
    args = build_args_from_dict(dic)

    assert args.arg1 == "value1"
    assert args.arg2 == 2
    assert args.arg3 == 0.3


if __name__ == "__main__":
    test_build_args_from_dict()
