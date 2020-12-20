import sys
from cogdl import options


def test_training_options():
    sys.argv = [sys.argv[0], "-t", "node_classification", "-m", "gcn", "-dt", "cora"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.task == "node_classification"
    assert args.model[0] == "gcn"
    assert args.dataset[0] == "cora"


def test_display_options():
    sys.argv = [sys.argv[0], "-dt", "cora"]
    parser = options.get_display_data_parser()
    args = parser.parse_args()
    print(args)

    assert args.dataset[0] == "cora"
    assert args.depth > 0


def test_download_options():
    sys.argv = [sys.argv[0], "-dt", "cora"]
    parser = options.get_download_data_parser()
    args = parser.parse_args()
    print(args)

    assert args.dataset[0] == "cora"


if __name__ == "__main__":
    test_training_options()
    test_display_options()
    test_download_options()
