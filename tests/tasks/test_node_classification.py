from cogdl import options
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    default_dict = {'hidden_size': 64,
                    'dropout': 0.5,
                    'patience': 100,
                    'max_epoch': 500,
                    'lr': 0.01,
                    'weight_decay': 5e-4}
    return build_args_from_dict(default_dict)

def test_gcn_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'gcn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] > 0.5

def test_gat_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'gat'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.alpha = 0.2
    args.nheads = 8
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] > 0.5

def test_mlp_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'mlp'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] > 0.4

if __name__ == "__main__":
    test_gcn_cora()
    test_gat_cora()
    test_mlp_cora()
