import cogdl.runner.options as options
from cogdl.runner.trainer import Trainer
from cogdl.runner.embed_trainer import EmbeddingTrainer
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper, EmbeddingModelWrapper
from cogdl.utils import set_random_seed


def raw_experiment(args, model_wrapper_args, data_wrapper_args):
    # setup dataset and specify `num_features` and `num_classes` for model
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    # setup model
    model = build_model(args)
    # specify configs for optimizer
    optimizer_cfg = dict(lr=args.lr, weight_decay=args.weight_decay)
    # setup model_wrapper
    if "embedding" in args.mw:
        model_wrapper = fetch_model_wrapper(args.mw)(model, **model_wrapper_args.__dict__)
    else:
        model_wrapper = fetch_model_wrapper(args.mw)(model, optimizer_cfg, **model_wrapper_args.__dict__)
    # setup data_wrapper
    dataset_wrapper = fetch_data_wrapper(args.dw)(dataset, **data_wrapper_args.__dict__)

    save_embedding_path = args.emb_path if hasattr(args, "emb_path") else None
    # setup trainer
    trainer = Trainer(max_epoch=args.max_epoch, device_ids=args.device_id, save_embedding_path=save_embedding_path)
    # Go!!!
    result = trainer.run(model_wrapper, dataset_wrapper)
    print(result)


def main():
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args, model_wrapper_args, data_wrapper_args = options.parse_args_and_arch(parser, args)
    args.dataset = args.dataset[0]
    print(args)
    set_random_seed(args.seed[0])
    raw_experiment(args, model_wrapper_args, data_wrapper_args)


if __name__ == "__main__":
    main()
