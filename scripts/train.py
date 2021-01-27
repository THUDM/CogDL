from cogdl import options, experiment


if __name__ == "__main__":
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    assert len(args.device_id) == 1

    experiment(task=args.task, dataset=args.dataset, model=args.model, args=args)
