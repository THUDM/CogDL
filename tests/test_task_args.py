import sys

from cogdl import options


def test_attributed_graph_clustering():
    sys.argv = [sys.argv[0], "-t", "attributed_graph_clustering", "-m", "prone", "-dt", "cora"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.task == "attributed_graph_clustering"
    assert args.model[0] == "prone"
    assert args.dataset[0] == "cora"
    assert args.num_clusters == 7


def test_graph_classification():
    sys.argv = [sys.argv[0], "-t", "graph_classification", "-m", "gin", "-dt", "mutag"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.task == "graph_classification"
    assert args.model[0] == "gin"
    assert args.dataset[0] == "mutag"
    assert args.degree_feature is False


def test_multiplex_link_prediction():
    sys.argv = [sys.argv[0], "-t", "multiplex_link_prediction", "-m", "gatne", "-dt", "amazon"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.task == "multiplex_link_prediction"
    assert args.model[0] == "gatne"
    assert args.dataset[0] == "amazon"
    assert args.eval_type == "all"


def test_link_prediction():
    sys.argv = [sys.argv[0], "-t", "link_prediction", "-m", "prone", "-dt", "ppi"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.task == "link_prediction"
    assert args.model[0] == "prone"
    assert args.dataset[0] == "ppi"
    assert args.evaluate_interval == 30


def test_unsupervised_graph_classification():
    sys.argv = [sys.argv[0], "-t", "unsupervised_graph_classification", "-m", "infograph", "-dt", "mutag"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.task == "unsupervised_graph_classification"
    assert args.model[0] == "infograph"
    assert args.dataset[0] == "mutag"
    assert args.num_shuffle == 10
    assert args.degree_feature is False


def test_unsupervised_node_classification():
    sys.argv = [sys.argv[0], "-t", "unsupervised_node_classification", "-m", "prone", "-dt", "ppi"]
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)

    assert args.task == "unsupervised_node_classification"
    assert args.model[0] == "prone"
    assert args.dataset[0] == "ppi"


if __name__ == "__main__":
    test_attributed_graph_clustering()
    test_graph_classification()
    test_multiplex_link_prediction()
    test_link_prediction()
    test_unsupervised_graph_classification()
    test_unsupervised_node_classification()
