import inspect

import torch

import cogdl.runner.options as options
from cogdl.runner.trainer import Trainer
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
from cogdl.utils import set_random_seed


def examine_link_prediction(args, dataset):
    if "link_prediction" in args.mw:
        args.num_entities = dataset.data.num_nodes
        # args.num_entities = len(torch.unique(self.data.edge_index))
        if dataset.data.edge_attr is not None:
            args.num_rels = len(torch.unique(dataset.data.edge_attr))
            args.monitor = "mrr"
        else:
            args.monitor = "auc"
    return args


def raw_experiment(args, model_wrapper_args, data_wrapper_args):
    # setup dataset and specify `num_features` and `num_classes` for model
    args.monitor = "val_acc"
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    if hasattr(args, "unsup") and args.unsup:
        args.num_classes = args.hidden_size
    else:
        args.num_classes = dataset.num_classes

    mw_class = fetch_model_wrapper(args.mw)
    dw_class = fetch_data_wrapper(args.dw)

    if mw_class is None:
        raise NotImplementedError("`model wrapper(--mw)` must be specified.")

    if dw_class is None:
        raise NotImplementedError("`data wrapper(--dw)` must be specified.")

    # unworthy code: share `args` between model and dataset_wrapper
    for key in inspect.signature(dw_class).parameters.keys():
        if hasattr(args, key) and key != "dataset":
            setattr(data_wrapper_args, key, getattr(args, key))
    # unworthy code: share `args` between model and model_wrapper
    for key in inspect.signature(mw_class).parameters.keys():
        if hasattr(args, key) and key != "model":
            setattr(model_wrapper_args, key, getattr(args, key))

    args = examine_link_prediction(args, dataset)

    # setup model
    model = build_model(args)
    # specify configs for optimizer
    optimizer_cfg = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, "hidden_size"):
        optimizer_cfg["hidden_size"] = args.hidden_size

    # setup model_wrapper
    if "embedding" in args.mw:
        model_wrapper = mw_class(model, **model_wrapper_args.__dict__)
    else:
        model_wrapper = mw_class(model, optimizer_cfg, **model_wrapper_args.__dict__)
    # setup data_wrapper
    dataset_wrapper = dw_class(dataset, **data_wrapper_args.__dict__)

    save_embedding_path = args.emb_path if hasattr(args, "emb_path") else None
    # setup trainer
    trainer = Trainer(
        max_epoch=args.max_epoch,
        device_ids=args.device_id,
        cpu=args.cpu,
        save_embedding_path=save_embedding_path,
        cpu_inference=args.cpu_inference,
        monitor=args.monitor,
    )
    # Go!!!
    result = trainer.run(model_wrapper, dataset_wrapper)
    print(result)


def main():
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args, model_wrapper_args, data_wrapper_args = options.parse_args_and_arch(parser, args)
    args.dataset = args.dataset[0]
    print(args)
    print(
        f""" 
   |----------------------------------------------------------------------------------|
    *** Using `{args.mw}` ModelWrapper and `{args.dw}` DataWrapper 
   |----------------------------------------------------------------------------------|"""
    )
    set_random_seed(args.seed[0])
    raw_experiment(args, model_wrapper_args, data_wrapper_args)


if __name__ == "__main__":
    main()
