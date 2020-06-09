from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict

def get_default_args():
    default_dict = {'hidden_size': 16,
                    'cpu': True,
                    'enhance': False,
                    'save_dir': ".",}
    return build_args_from_dict(default_dict)

def test_metapath2vec_gtn_acm():
    args = get_default_args()
    args.task = 'multiplex_node_classification'
    args.dataset = 'gtn-acm'
    args.model = 'metapath2vec'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.schema = "No"
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] > 0
    

def test_metapath2vec_gtn_imdb():
    args = get_default_args()
    args.task = 'multiplex_node_classification'
    args.dataset = 'gtn-imdb'
    args.model = 'metapath2vec'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.schema = "No"
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] > 0
    

def test_pte_gtn_imdb():
    args = get_default_args()
    args.task = 'multiplex_node_classification'
    args.dataset = 'gtn-imdb'
    args.model = 'pte'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 10
    args.alpha = 0.025
    args.order = "No"
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] > 0


def test_pte_gtn_dblp():
    args = get_default_args()
    args.task = 'multiplex_node_classification'
    args.dataset = 'gtn-dblp'
    args.model = 'pte'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 10
    args.alpha = 0.025
    args.order = "No"
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] > 0

def test_hin2vec_dblp():
    args = get_default_args()
    args.task = 'multiplex_node_classification'
    args.dataset = 'gtn-dblp'
    args.model = 'hin2vec'
    dataset = build_dataset(args)
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 1000
    args.hop = 2
    args.epoches = 1
    args.lr = 0.025
    args.cpu = True
    model = build_model(args)
    task = build_task(args)
    ret = task.train()
    assert ret['f1'] > 0

if __name__ == "__main__":
    test_metapath2vec_gtn_acm()
    test_metapath2vec_gtn_imdb()
    test_pte_gtn_imdb()
    test_pte_gtn_dblp()
    test_hin2vec_dblp()
