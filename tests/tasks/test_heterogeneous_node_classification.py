import torch
from cogdl import options
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {'hidden_size': 8,
                    'dropout': 0.5,
                    'patience': 1,
                    'max_epoch': 1,
                    'cpu': not cuda_available,
                    'lr': 0.01,
                    'weight_decay': 5e-4}
    return build_args_from_dict(default_dict)

def test_gtn_gtn_imdb():
    args = get_default_args()
    args.task = 'heterogeneous_node_classification'
    args.dataset = 'gtn-imdb'
    args.model = 'gtn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_edge = dataset.num_edge
    args.num_nodes = dataset.num_nodes
    args.num_channels = 2
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] >= 0 and ret['f1'] <= 1

def test_han_gtn_acm():
    args = get_default_args()
    args.task = 'heterogeneous_node_classification'
    args.dataset = 'gtn-acm'
    args.model = 'han'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_edge = dataset.num_edge
    args.num_nodes = dataset.num_nodes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] >= 0 and ret['f1'] <= 1

def test_han_gtn_dblp():
    args = get_default_args()
    args.task = 'heterogeneous_node_classification'
    args.dataset = 'gtn-dblp'
    args.model = 'han'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_edge = dataset.num_edge
    args.num_nodes = dataset.num_nodes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] >= 0 and ret['f1'] <= 1

def test_han_han_imdb():
    args = get_default_args()
    args.task = 'heterogeneous_node_classification'
    args.dataset = 'han-imdb'
    args.model = 'han'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_edge = dataset.num_edge
    args.num_nodes = dataset.num_nodes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] >= 0 and ret['f1'] <= 1

def test_han_han_acm():
    args = get_default_args()
    args.task = 'heterogeneous_node_classification'
    args.dataset = 'han-acm'
    args.model = 'han'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_edge = dataset.num_edge
    args.num_nodes = dataset.num_nodes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] >= 0 and ret['f1'] <= 1

def test_han_han_dblp():
    args = get_default_args()
    args.task = 'heterogeneous_node_classification'
    args.dataset = 'han-dblp'
    args.model = 'han'
    args.cpu = True
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_edge = dataset.num_edge
    args.num_nodes = dataset.num_nodes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] >= 0 and ret['f1'] <= 1

if __name__ == "__main__":
    test_gtn_gtn_imdb()
    test_han_gtn_acm()
    test_han_gtn_dblp()
    test_han_han_imdb()
    test_han_han_acm()
    test_han_han_dblp()
