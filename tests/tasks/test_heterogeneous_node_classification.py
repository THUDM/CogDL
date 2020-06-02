from cogdl import options
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    default_dict = {'hidden_size': 64,
                    'dropout': 0.5,
                    'patience': 1,
                    'max_epoch': 1,
                    'cpu': True,
                    'lr': 0.01,
                    'weight_decay': 5e-4}
    return build_args_from_dict(default_dict)

def test_gtn_imdb():
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
    args.num_layers = 3
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] >= 0 and ret['f1'] <= 1

if __name__ == "__main__":
    test_gtn_imdb()
