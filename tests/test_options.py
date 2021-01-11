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


def test_get_default_args():
    args = options.get_default_args(
        task="node_classification", dataset=["cora", "citeseer"], model=["gcn", "gat"], hidden_size=128
    )

    assert args.task == "node_classification"
    assert args.model[0] == "gcn"
    assert args.model[1] == "gat"
    assert args.dataset[0] == "cora"
    assert args.dataset[1] == "citeseer"
    assert args.hidden_size == 128


def test_get_task_model_args():
    args = options.get_task_model_args(task="node_classification", model="gcn")
    assert args.lr == 0.01
    assert args.weight_decay == 5e-4
    assert args.dropout == 0.5


if __name__ == "__main__":
    test_training_options()
    test_display_options()
    test_download_options()
    test_get_default_args()
