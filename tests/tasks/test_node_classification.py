import torch
from cogdl import options
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    cuda_available = torch.cuda.is_available()
    default_dict = {'hidden_size': 16,
                    'dropout': 0.5,
                    'patience': 2,
                    'max_epoch': 3,
                    'cpu': not cuda_available,
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
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

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
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_mlp_pubmed():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'pubmed'
    args.model = 'mlp'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_mixhop_citeseer():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'citeseer'
    args.model = 'mixhop'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_graphsage_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'graphsage'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    args.hidden_size = [128]
    args.sample_size = [10, 10]
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_pyg_cheb_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'chebyshev'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    args.filter_size = 5
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_pyg_gcn_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'pyg_gcn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_pyg_gat_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'pyg_gat'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_heads = 8
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_pyg_infomax_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'infomax'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_pyg_unet_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'unet'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_pyg_drgcn_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'drgcn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 2
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_pyg_drgat_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'drgat'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_heads = 8
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_disengcn_cora():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'disengcn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.K = [4, 2]
    args.activation = "leaky_relu"
    args.tau = 1.0
    args.iterations = 3
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

def test_graph_mix():
    args = get_default_args()
    args.task = 'node_classification'
    args.dataset = 'cora'
    args.model = 'gcnmix'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.rampup_starts = 1
    args.rampup_ends = 100
    args.mixup_consistency = 5.0
    args.ema_decay = 0.999
    args.alpha = 1.0
    args.temperature = 1.0
    args.k = 10
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

if __name__ == "__main__":
    test_gcn_cora()
    test_gat_cora()
    test_mlp_pubmed()
    test_mixhop_citeseer()
    test_graphsage_cora()
    test_pyg_cheb_cora()
    test_pyg_gcn_cora()
    test_pyg_gat_cora()
    test_pyg_infomax_cora()
    test_pyg_unet_cora()
    test_pyg_drgcn_cora()
    test_pyg_drgat_cora()
    test_disengcn_cora()
    test_graph_mix()
