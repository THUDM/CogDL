import sys

from cogdl import options


def test_attributed_graph_clustering():
    sys.argv = [sys.argv[0], "-m", "daegc", "-dt", "cora"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.model[0] == "daegc"
    assert args.dataset[0] == "cora"
    assert args.num_clusters == 7


def test_graph_classification():
    sys.argv = [sys.argv[0], "-m", "dgk", "-dt", "mutag"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.model[0] == "dgk"
    assert args.dataset[0] == "mutag"


def test_multiplex_link_prediction():
    sys.argv = [sys.argv[0], "-m", "gatne", "-dt", "amazon"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.model[0] == "gatne"
    assert args.dataset[0] == "amazon"
    assert args.eval_type == "all"


def test_link_prediction():
    sys.argv = [sys.argv[0], "-m", "prone", "-dt", "ppi"]
    sys.argv += ["--mw", "embedding_link_prediction_mw", "--dw", "embedding_link_prediction_dw"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.model[0] == "prone"
    assert args.dataset[0] == "ppi"
    assert args.mw == "embedding_link_prediction_mw"
    assert args.dw == "embedding_link_prediction_dw"


def test_unsupervised_graph_classification():
    sys.argv = [sys.argv[0], "-m", "infograph", "-dt", "mutag"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.model[0] == "infograph"
    assert args.dataset[0] == "mutag"


def test_unsupervised_node_classification():
    sys.argv = [sys.argv[0], "-m", "prone", "-dt", "ppi"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.model[0] == "prone"
    assert args.dataset[0] == "ppi"


if __name__ == "__main__":
    test_attributed_graph_clustering()
    test_graph_classification()
    test_multiplex_link_prediction()
    test_link_prediction()
    test_unsupervised_graph_classification()
    test_unsupervised_node_classification()
