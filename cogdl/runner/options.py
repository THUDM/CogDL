import sys
import argparse
import copy
from typing import Optional

from cogdl.datasets import try_import_dataset
from cogdl.models import MODEL_REGISTRY, try_import_model
from cogdl.wrappers import fetch_data_wrapper, fetch_model_wrapper
from cogdl.utils import build_args_from_dict
from cogdl.wrappers.default_match import get_wrappers_name


def add_args(args: list):
    parser = argparse.ArgumentParser()
    if "lr" in args:
        parser.add_argument("--lr", default=0.01, type=float)
    if "max_epoch" in args:
        parser.add_argument("--max-epoch", default=500, type=int)


def add_arguments(args: list):
    parser = argparse.ArgumentParser()
    for item in args:
        name, _type, default = item
        parser.add_argument(f"--{name}", default=default, type=_type)
    return parser


def get_parser():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    # fmt: off
    # parser.add_argument("--log-interval", type=int, default=1000, metavar="N",
    #                     help="log progress every N batches (when progress bar is disabled)")
    # parser.add_argument("--tensorboard-logdir", metavar="DIR", default='',
    #                     help="path to save logs for tensorboard, should match --logdir "
    #                          "of running tensorboard (default: no tensorboard logging)")
    parser.add_argument("--seed", default=[1], type=int, nargs="+", metavar="N",
                        help="pseudo random number generator seed")
    parser.add_argument("--max-epoch", default=500, type=int)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--cpu", action="store_true", help="use CPU instead of CUDA")
    parser.add_argument("--device-id", default=[0], type=int, nargs="+",
                        help="which GPU to use")
    parser.add_argument("--save-dir", default=".", type=str)
    parser.add_argument("--checkpoint", type=str, default=None, help="load pre-trained model")
    parser.add_argument("--save-model", type=str, default=None, help="save trained model")
    parser.add_argument("--use-best-config", action="store_true", help="use best config")
    parser.add_argument("--unsup", action="store_true")
    parser.add_argument("--cpu-inference", action="store_true", help="do validation and test in cpu")
    parser.add_argument("--monitor", type=str, default="val_acc")
    parser.add_argument("--progress-bar", type=str, default="epoch")
    parser.add_argument("--n-warmup-steps", type=int, default=10000)

    # fmt: on
    return parser


def add_data_wrapper_args(parser):
    group = parser.add_argument_group("Data warpper configuration")
    # fmt: off
    group.add_argument("--dw", "-t", type=str, default=None, metavar="DWRAPPER", required=False,
                       help="Data Wrapper")
    # fmt: on
    return group


def add_model_wrapper_args(parser):
    group = parser.add_argument_group("Trainer configuration")
    # fmt: off
    group.add_argument("--mw", type=str, default=None, metavar="MWRAPPER", required=False,
                       help="Model Wrapper")
    # fmt: on
    return group


def add_dataset_args(parser):
    group = parser.add_argument_group("Dataset and data loading")
    # fmt: off
    group.add_argument("--dataset", "-dt", metavar="DATASET", nargs="+", required=True,
                       help="Dataset")
    # fmt: on
    return group


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off
    group.add_argument("--model", "-m", metavar="MODEL", required=True,
                       help="Model Architecture")
    # fmt: on
    return group


def get_training_parser():
    parser = get_parser()
    add_dataset_args(parser)
    add_model_args(parser)
    add_data_wrapper_args(parser)
    add_model_wrapper_args(parser)
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

    # The parser doesn"t know about specific args, so we parse twice.
    parser = get_training_parser()
    args, _ = parser.parse_known_args()
    args = parse_args_and_arch(parser, args)
    for key, value in kwargs.items():
        args.__setattr__(key, value)
    return args


def get_diff_args(args1, args2):
    d1 = copy.deepcopy(args1.__dict__)
    d2 = args2.__dict__
    for k in d2.keys():
        d1.pop(k, None)
    return build_args_from_dict(d1)


def parse_args_and_arch(parser, args):
    # Add *-specific args to parser.
    # basic_args = copy.deepcopy(args)
    if try_import_model(args.model):
        MODEL_REGISTRY[args.model].add_args(parser)
    args_m, _ = parser.parse_known_args()
    # model_args = get_diff_args(args_m, basic_args)

    for dataset in args.dataset:
        try_import_dataset(dataset)
        # if hasattr(DATASET_REGISTRY[dataset], "add_args"):
        #     DATASET_REGISTRY[dataset].add_args(parser)

    default_wrappers = get_wrappers_name(args.model)
    if default_wrappers is not None:
        mw, dw = default_wrappers
    else:
        mw, dw = None, None

    if args.dw is not None:
        dw = args.dw
    if hasattr(fetch_data_wrapper(dw), "add_args"):
        fetch_data_wrapper(dw).add_args(parser)

    args_dw, _ = parser.parse_known_args()

    data_wrapper_args = get_diff_args(args_dw, args_m)

    if args.mw is not None:
        mw = args.mw
    if hasattr(fetch_model_wrapper(mw), "add_args"):
        fetch_model_wrapper(mw).add_args(parser)
    args_mw, _ = parser.parse_known_args()
    model_wrapper_args = get_diff_args(args_mw, args_dw)

    # Parse a second time.
    args = parser.parse_args()
    args.mw = mw
    args.dw = dw
    # return args
    return args, model_wrapper_args, data_wrapper_args


def get_model_args(task, model=None):
    sys.argv = [sys.argv[0], "-t", task, "-m"] + ["gcn"] + ["-dt"] + ["cora"]
    parser = get_training_parser()
    if model is not None:
        if try_import_model(model):
            MODEL_REGISTRY[model].add_args(parser)
    args = parser.parse_args()
    args.task = task
    if model is not None:
        args.model = model
    return args
