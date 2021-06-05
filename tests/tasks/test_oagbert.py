from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict
import torch

def get_default_args():
    default_dict = {
        "save_dir": "./embedding",
        "checkpoint": False,
        "device_id": [0],
        "fast_spmm": False,
        "token_type": 'FOS',
        'wabs': False, 
        'weight_decay': 0.0005, 
        'wprop': False,
        'include_fields': ['title'],
        'freeze': False,
        'testing': True
    }
    return build_args_from_dict(default_dict)

def zero_shot_infer_arxiv():
    args = get_default_args()
    args.task = 'zero_shot_infer'
    args.dataset = 'arxivvenue'
    args.model = 'oagbert'
    args.cuda = [i for i in range(torch.cuda.device_count())]
    task = build_task(args)
    ret = task.train()
    assert ret['Accuracy'] < 1

def finetune_arxiv():
    args = get_default_args()
    args.task = 'supervised_classification'
    args.dataset = 'arxivvenue'
    args.model = 'oagbert'
    args.cuda = 6
    task = build_task(args)
    ret = task.train()
    assert ret['Accuracy'] < 1


if __name__ == '__main__':
    zero_shot_infer_arxiv()
    finetune_arxiv()