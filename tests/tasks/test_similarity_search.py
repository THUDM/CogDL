import torch
from cogdl.tasks import build_task
from cogdl.utils import build_args_from_dict


def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {
        "hidden_size": 64,
        "device_id": [0],
        "max_epoch": 1,
        "load_path": "./saved/gcc_pretrained.pth",
        "cpu": not cuda_available,
        "checkpoint": False,
    }
    return build_args_from_dict(default_dict)


def test_gcc_kdd_icdm():
    args = get_default_args()
    args.task = "similarity_search"
    args.dataset = "kdd_icdm"
    args.model = "gcc"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Recall @ 20"] <= 1


def test_gcc_sigir_cikm():
    args = get_default_args()
    args.task = "similarity_search"
    args.dataset = "sigir_cikm"
    args.model = "gcc"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Recall @ 20"] <= 1


def test_gcc_sigmod_icde():
    args = get_default_args()
    args.task = "similarity_search"
    args.dataset = "sigmod_icde"
    args.model = "gcc"
    task = build_task(args)
    ret = task.train()
    assert 0 <= ret["Recall @ 20"] <= 1


if __name__ == "__main__":
    test_gcc_kdd_icdm()
    test_gcc_sigir_cikm()
    test_gcc_sigmod_icde()
