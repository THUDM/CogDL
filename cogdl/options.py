import sys
import argparse

from cogdl.datasets import DATASET_REGISTRY, try_import_dataset
from cogdl.models import MODEL_REGISTRY, try_import_model
from cogdl.tasks import TASK_REGISTRY, try_import_task
from cogdl.trainers import TRAINER_REGISTRY, try_import_trainer


def get_parser():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    # fmt: off
    # parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
    #                     help='log progress every N batches (when progress bar is disabled)')
    # parser.add_argument('--tensorboard-logdir', metavar='DIR', default='',
    #                     help='path to save logs for tensorboard, should match --logdir '
    #                          'of running tensorboard (default: no tensorboard logging)')
    parser.add_argument('--seed', default=[1], type=int, nargs='+', metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--max-epoch', default=500, type=int)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--device-id', default=[0], type=int, nargs='+',
                        help='which GPU to use')
    parser.add_argument('--save-dir', default='.', type=str)
    parser.add_argument('--checkpoint', type=str, default=None, help='load pre-trained model')
    parser.add_argument('--save-model', type=str, default=None, help='save trained model')
    parser.add_argument('--use-best-config', action='store_true', help='use best config')
    parser.add_argument("--actnn", action="store_true")

    # fmt: on
    return parser


def add_task_args(parser):
    group = parser.add_argument_group("Task configuration")
    # fmt: off
    group.add_argument('--task', '-t', default='node_classification', metavar='TASK', required=True,
                       help='Task')
    # fmt: on
    return group


def add_dataset_args(parser):
    group = parser.add_argument_group("Dataset and data loading")
    # fmt: off
    group.add_argument('--dataset', '-dt', metavar='DATASET', nargs='+', required=True,
                       help='Dataset')
    # fmt: on
    return group


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off
    group.add_argument('--model', '-m', metavar='MODEL', nargs='+', required=True,
                       help='Model Architecture')
    group.add_argument('--fast-spmm', action="store_true", required=False,
                       help='whether to use gespmm')
    # fmt: on
    return group


def add_trainer_args(parser):
    group = parser.add_argument_group("Trainer configuration")
    # fmt: off
    group.add_argument('--trainer', metavar='TRAINER', required=False,
                       help='Trainer')
    # fmt: on
    return group


def get_training_parser():
    parser = get_parser()
    add_task_args(parser)
    add_dataset_args(parser)
    add_model_args(parser)
    add_trainer_args(parser)
    return parser


def get_display_data_parser():
    parser = get_parser()
    add_dataset_args(parser)
    parser.add_argument("--depth", default=3, type=int)

    return parser


def get_download_data_parser():
    parser = get_parser()
    add_dataset_args(parser)

    return parser


def get_default_args(task: str, dataset, model, **kwargs):
    if not isinstance(dataset, list):
        dataset = [dataset]
    if not isinstance(model, list):
        model = [model]
    sys.argv = [sys.argv[0], "-t", task, "-m"] + model + ["-dt"] + dataset

    # The parser doesn't know about specific args, so we parse twice.
    parser = get_training_parser()
    args, _ = parser.parse_known_args()
    args = parse_args_and_arch(parser, args)
    for key, value in kwargs.items():
        args.__setattr__(key, value)
    return args


def parse_args_and_arch(parser, args):
    # Add *-specific args to parser.
    try_import_task(args.task)
    TASK_REGISTRY[args.task].add_args(parser)
    for model in args.model:
        if try_import_model(model):
            MODEL_REGISTRY[model].add_args(parser)
    for dataset in args.dataset:
        if try_import_dataset(dataset):
            if hasattr(DATASET_REGISTRY[dataset], "add_args"):
                DATASET_REGISTRY[dataset].add_args(parser)

    if "trainer" in args and args.trainer is not None:
        if try_import_trainer(args.trainer):
            if hasattr(TRAINER_REGISTRY[args.trainer], "add_args"):
                TRAINER_REGISTRY[args.trainer].add_args(parser)
    else:
        for model in args.model:
            tr = MODEL_REGISTRY[model].get_trainer(args)
            if tr is not None:
                tr.add_args(parser)
    # Parse a second time.
    args = parser.parse_args()
    return args


def get_task_model_args(task, model=None):
    sys.argv = [sys.argv[0], "-t", task, "-m"] + ["gcn"] + ["-dt"] + ["cora"]
    parser = get_training_parser()
    try_import_task(task)
    TASK_REGISTRY[task].add_args(parser)
    if model is not None:
        if try_import_model(model):
            MODEL_REGISTRY[model].add_args(parser)
    args = parser.parse_args()
    args.task = task
    if model is not None:
        args.model = model
    return args
