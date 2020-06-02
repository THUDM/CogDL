from cogdl import options
from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    default_dict = {'hidden_size': 64,
                    'dropout': 0.5,
                    'patience': 1,
                    'max_epoch': 2,
                    'cpu': True,
                    'lr': 0.01,
                    'weight_decay': 5e-4}
    return build_args_from_dict(default_dict)

def test_fastgcn_cora():
    args = get_default_args()
    args.task = 'node_classification_sampling'
    args.dataset = 'cora'
    args.model = 'fastgcn'
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_layers = 3
    args.sample_size = [512,256,256]
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['Acc'] >= 0 and ret['Acc'] <= 1

if __name__ == "__main__":
    test_fastgcn_cora()
