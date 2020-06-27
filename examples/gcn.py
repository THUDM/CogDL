import torch
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {'hidden_size': 16,
                    'dropout': 0.5,
                    'patience': 100,
                    'max_epoch': 500,
                    'cpu': not cuda_available,
                    'lr': 0.01,
                    'weight_decay': 5e-4}
    return build_args_from_dict(default_dict)


if __name__ == "__main__":
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'pyg_gcn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args, dataset=dataset, model=model)
    ret = task.train()
