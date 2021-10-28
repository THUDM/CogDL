from cogdl import options, experiment


if __name__ == "__main__":
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)

    experiment(dataset=args.dataset, model=args.model, args=args)
