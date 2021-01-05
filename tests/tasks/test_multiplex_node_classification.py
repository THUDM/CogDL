from cogdl.tasks import build_task
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from cogdl.utils import build_args_from_dict


def get_default_args():
    default_dict = {
        "hidden_size": 16,
        "cpu": True,
        "enhance": False,
        "save_dir": "./embedding",
        "checkpoint": False,
        "device_id": [0],
    }
    return build_args_from_dict(default_dict)


# def add_args_for_gcc(args):
#     args.load_path = "./saved/gcc_pretrained.pth"
#     return args

# def test_gcc_imdb():
#     args = get_default_args()
#     args = add_args_for_gcc(args)
#     args.task = 'multiplex_node_classification'
#     args.dataset = 'gtn-imdb'
#     args.model = 'gcc'
#     dataset = build_dataset(args)
#     args.num_features = dataset.num_features
#     args.num_classes = dataset.num_classes
#     args.num_edge = dataset.num_edge
#     args.num_nodes = dataset.num_nodes
#     args.num_channels = 2
#     args.num_layers = 2
#     model = build_model(args)
#     task = build_task(args)
#     ret = task.train()
#     assert ret['f1'] >= 0 and ret['f1'] <= 1

# def test_gcc_acm():
#     args = get_default_args()
#     args = add_args_for_gcc(args)
#     args.task = 'multiplex_node_classification'
#     args.dataset = 'gtn-acm'
#     args.model = 'gcc'
#     dataset = build_dataset(args)
#     args.num_features = dataset.num_features
#     args.num_classes = dataset.num_classes
#     args.num_edge = dataset.num_edge
#     args.num_nodes = dataset.num_nodes
#     args.num_channels = 2
#     args.num_layers = 2
#     model = build_model(args)
#     task = build_task(args)
#     ret = task.train()
#     assert ret['f1'] >= 0 and ret['f1'] <= 1

# def test_gcc_dblp():
#     args = get_default_args()
#     args = add_args_for_gcc(args)
#     args.task = 'multiplex_node_classification'
#     args.dataset = 'gtn-dblp'
#     args.model = 'gcc'
#     dataset = build_dataset(args)
#     args.num_features = dataset.num_features
#     args.num_classes = dataset.num_classes
#     args.num_edge = dataset.num_edge
#     args.num_nodes = dataset.num_nodes
#     args.num_channels = 2
#     args.num_layers = 2
#     model = build_model(args)
#     task = build_task(args)
#     ret = task.train()
#     assert ret['f1'] >= 0 and ret['f1'] <= 1


def test_metapath2vec_gtn_acm():
    args = get_default_args()
    args.task = "multiplex_node_classification"
    args.dataset = "gtn-acm"
    args.model = "metapath2vec"
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.schema = "No"
    task = build_task(args)
    ret = task.train()
    assert ret["f1"] > 0


def test_metapath2vec_gtn_imdb():
    args = get_default_args()
    args.task = "multiplex_node_classification"
    args.dataset = "gtn-imdb"
    args.model = "metapath2vec"
    args.walk_length = 5
    args.walk_num = 1
    args.window_size = 3
    args.worker = 5
    args.iteration = 1
    args.schema = "No"
    task = build_task(args)
    ret = task.train()
    assert ret["f1"] > 0


def test_pte_gtn_imdb():
    args = get_default_args()
    args.task = "multiplex_node_classification"
    args.dataset = "gtn-imdb"
    args.model = "pte"
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 10
    args.alpha = 0.025
    args.order = "No"
    task = build_task(args)
    ret = task.train()
    assert ret["f1"] > 0


def test_pte_gtn_dblp():
    args = get_default_args()
    args.task = "multiplex_node_classification"
    args.dataset = "gtn-dblp"
    args.model = "pte"
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 10
    args.alpha = 0.025
    args.order = "No"
    task = build_task(args)
    ret = task.train()
    assert ret["f1"] > 0


def test_hin2vec_dblp():
    args = get_default_args()
    args.task = "multiplex_node_classification"
    args.dataset = "gtn-dblp"
    args.model = "hin2vec"
    args.walk_length = 5
    args.walk_num = 1
    args.negative = 3
    args.batch_size = 1000
    args.hop = 2
    args.epochs = 1
    args.lr = 0.025
    args.cpu = True
    task = build_task(args)
    ret = task.train()
    assert ret["f1"] > 0


if __name__ == "__main__":
    test_metapath2vec_gtn_acm()
    test_metapath2vec_gtn_imdb()
    test_pte_gtn_imdb()
    test_pte_gtn_dblp()
    test_hin2vec_dblp()
