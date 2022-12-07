import sys
import argparse
import copy
import warnings

from cogdl.datasets import try_adding_dataset_args
from cogdl.models import try_adding_model_args
from cogdl.wrappers import fetch_data_wrapper, fetch_model_wrapper
from cogdl.utils import build_args_from_dict
from cogdl.wrappers.default_match import get_wrappers_name


def get_parser():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    # fmt: off
    parser.add_argument("--seed", default=[1], type=int, nargs="+", metavar="N",
                        help="pseudo random number generator seed")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--max-epoch", default=None, type=int)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--n-warmup-steps", type=float, default=0.)
    parser.add_argument("--split", default=[0], type=int, nargs="+", metavar="N")
    parser.add_argument("--clip-grad-norm", default=5., type=float)

    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/model.pt", help="path to save model")
    parser.add_argument("--save-emb-path", type=str, default=None, help="path to save embeddings")
    parser.add_argument("--load-emb-path", type=str, default=None, help="path to load embeddings")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=".", help="path to save logs")
    parser.add_argument("--project", type=str, default="cogdl-exp", help="project name for wandb")

    parser.add_argument("--use-best-config", action="store_true", help="use best config")
    parser.add_argument("--unsup", action="store_true")
    parser.add_argument("--nstage", type=int, default=1)
    parser.add_argument("--eval-step", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=3)

    parser.add_argument("--devices", default=[0], type=int, nargs="+", help="which GPU to use")
    parser.add_argument("--cpu", action="store_true", help="use CPU instead of CUDA")
    parser.add_argument("--cpu-inference", action="store_true", help="do validation and test in cpu")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--progress-bar", type=str, default="epoch")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=13425)
    parser.add_argument("--master-addr", type=str, default="localhost")

    parser.add_argument("--return_model", action="store_true")

    parser.add_argument("--actnn", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--rp-ratio", type=int, default=1)
    parser.add_argument("--do_test", default=True)
    parser.add_argument("--do_valid", default=True)
    # fmt: on
    return parser


def add_data_wrapper_args(parser):
    group = parser.add_argument_group("Data wrapper configuration")
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
    group.add_argument("--model", "-m", default="autognn", metavar="MODEL", nargs="+", required=False,
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


def get_default_args(dataset, model, **kwargs):
    if not isinstance(dataset, list):
        dataset = [dataset]
    if not isinstance(model, list):
        model = [model]
    sys.argv = [sys.argv[0], "-m"] + model + ["-dt"] + dataset
    if "mw" in kwargs and kwargs["mw"] is not None:
        sys.argv += ["--mw"] + [kwargs["mw"]]
    if "dw" in kwargs and kwargs["dw"] is not None:
        sys.argv += ["--dw"] + [kwargs["dw"]]

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
    for model in args.model:
        try_adding_model_args(model, parser)

    for dataset in args.dataset:
        try_adding_dataset_args(dataset, parser)

    if len(args.model) > 1:
        warnings.warn("Please ensure that models could use the same model wrapper!")
    default_wrappers = get_wrappers_name(args.model[0])
    if default_wrappers is not None:
        mw, dw = default_wrappers
    else:
        mw, dw = None, None

    if args.dw is not None:
        dw = args.dw
    if dw is None:
        # warnings.warn("Using default data wrapper ('node_classification_dw') for training!")
        dw = "node_classification_dw"
    if hasattr(fetch_data_wrapper(dw), "add_args"):
        fetch_data_wrapper(dw).add_args(parser)

    if args.mw is not None:
        mw = args.mw
    if mw is None:
        # warnings.warn("Using default model wrapper ('node_classification_mw') for training!")
        mw = "node_classification_mw"
    if hasattr(fetch_model_wrapper(mw), "add_args"):
        fetch_model_wrapper(mw).add_args(parser)

    # Parse a second time.
    args = parser.parse_args()
    args.mw = mw
    args.dw = dw
    return args
