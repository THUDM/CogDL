BEST_CONFIGS = {
    "node_classification": {
        "chebyshev": {"general": {}},
        "dropedge_gcn": {"general": {}},
        "gat": {
            "general": {"lr": 0.005, "max_epoch": 1000},
            "citeseer": {"weight_decay": 0.001},
            "pubmed": {"weight_decay": 0.001},
        },
        "gcn": {"general": {}},
        "gcnii": {
            "general": {
                "max_epoch": 1000,
                "dropout": 0.5,
                "wd1": 0.001,
                "wd2": 5e-4,
            },
            "cora": {
                "num_layers": 64,
                "hidden_size": 64,
                "dropout": 0.6,
            },
            "citeseer": {
                "num_layers": 32,
                "hidden_size": 256,
                "lr": 0.001,
                "patience": 200,
                "max_epoch": 2000,
                "lmbda": 0.6,
                "dropout": 0.7,
            },
            "pubmed": {
                "num_layers": 16,
                "hidden_size": 256,
                "lmbda": 0.4,
                "dropout": 0.5,
                "wd1": 5e-4,
            },
        },
        "gdc_gcn": {
            "general": {"hidden_size": 16},
        },
        "grand": {
            "general": {
                "max_epoch": 1000,
            },
            "cora": {
                "order": 8,
                "sample": 4,
                "lam": 1.0,
                "tem": 0.5,
                "alpha": 0.5,
                "patience": 200,
                "input_dropout": 0.5,
                "hidden_dropout": 0.5,
            },
            "citeseer": {
                "order": 2,
                "sample": 2,
                "lam": 0.7,
                "tem": 0.3,
                "alpha": 0.5,
                "input_dropout": 0.0,
                "hidden_dropout": 0.2,
                "patience": 200,
            },
            "pubmed": {
                "order": 5,
                "sample": 4,
                "lam": 1.0,
                "tem": 0.2,
                "alpha": 0.5,
                "lr": 0.2,
                "bn": True,
                "input_dropout": 0.6,
                "hidden_dropout": 0.8,
            },
        },
        "graphsage": {
            "general": {},
        },
        "sgc": {
            "general": {
                "hidden_size": 16,
                "dropout": 0.5,
            },
        },
        "sgcpn": {
            "general": {
                "lr": 0.005,
                "max_epoch": 1000,
                "patience": 1000,
                "norm_mode": "PN",
                "norm_scale": 10,
                "dropout": 0.6,
            },
        },
        "sign": {
            "general": {
                "lr": 0.00005,
                "hidden_size": 2048,
                "dropout": 0.5,
                "dropedge_rate": 0.2,
            },
        },
        "srgcn": {
            "general": {
                "lr": 0.005,
                "max_epoch": 1000,
            },
            "cora": {"dropout": 0.6},
            "citeseer": {"dropout": 0.6},
        },
        "unet": {
            "general": {
                "max_epoch": 1000,
                "n_dropout": 0.90,
                "adj_dropout": 0.05,
                "hidden_size": 128,
                "aug_adj": False,
                "improved": False,
                "n_pool": 4,
                "pool_rate": [0.7, 0.5, 0.5, 0.4],
            },
        },
    },
    "unsupervised_node_classification": {
        "deepwalk": {
            "general": {},
        },
        "dngr": {
            "general": {
                "hidden_size": 128,
                "lr": 0.001,
                "max_epoch": 500,
                "hidden_size1": 1000,
                "hidden_size2": 128,
                "noise": 0.2,
                "alpha": 0.1,
                "step": 10,
            },
        },
        "grarep": {
            "general": {},
        },
        "hope": {
            "general": {},
        },
        "line": {
            "general": {},
            "blogcatalog": {"walk_num": 40},
        },
        "netmf": {
            "general": {},
            "ppi-ne": {"window_size": 10, "is_large": True},
            "blogcatalog": {"window_size": 10, "is_large": True},
            "wikipedia": {"window_size": 1},
        },
        "netsmf": {
            "general": {"window_size": 5},
        },
        "node2vec": {
            "general": {},
        },
        "prone": {
            "general": {"step": 10},
            "ppi-ne": {"mu": 0.0},
            "wikipedia": {"mu": -4.0},
        },
        "sdne": {
            "general": {},
        },
        "spectral": {
            "general": {},
        },
        "dgi": {
            "general": {"weight_decay": 0},
        },
        "gcc": {
            "general": {},
        },
        "grace": {
            "general": {
                "weight_decay": 0,
                "max_epoch": 1000,
                "patience": 20,
            },
            "cora": {
                "lr": 0.0005,
                "weight_decay": 0.00001,
                "tau": 0.4,
                "drop_feature_rates": [0.3, 0.4],
                "drop_edge_rates": [0.2, 0.4],
                "max_epoch": 200,
                "hidden_size": 128,
                "proj_hidden_size": 128,
            },
            "citeseer": {
                "hidden_size": 256,
                "proj_hidden_size": 256,
                "drop_feature_rates": [0.3, 0.2],
                "drop_edge_rates": [0.2, 0.0],
                "lr": 0.001,
                "_weight_decay": 0.00001,
                "tau": 0.9,
                "activation": "prelu",
            },
            "pubmed": {
                "hidden_size": 256,
                "proj_hidden_size": 256,
                "drop_edge_rates": [0.4, 0.1],
                "drop_feature_rates": [0.0, 0.2],
                "tau": 0.7,
                "lr": 0.001,
                "weight_decay": 0.00001,
            },
        },
        "unsup_graphsage": {
            "lr": 0.001,
            "weight_decay": 0,
            "max_epoch": 3000,
        },
    },
    "graph_classification": {
        "gin": {
            "general": {"lr": 0.001},
            "imdb-b": {"degree_feature": True},
            "imdb-m": {"degree_feature": True},
            "collab": {"degree_feature": True},
            "proteins": {
                "num_layers": 5,
                "dropout": 0.0,
            },
            "nci1": {
                "num_layers": 5,
                "dropout": 0.3,
                "hidden_size": 64,
            },
        },
        "infograph": {
            "general": {
                "lr": 0.0001,
                "weight_decay": 5e-4,
                "sup": False,
            },
            "mutag": {
                "num_layers": 1,
                "epoch": 20,
            },
            "imdb-b": {"degree_feature": True},
            "imdb-m": {"degree_feature": True},
            "collab": {"degree_feature": True},
            "nci1": {"num_layers": 3},
        },
        "sortpool": {
            "nci1": {
                "dropout": 0.3,
                "hidden_size": 64,
                "num_layers": 5,
            },
        },
        "patchy_san": {
            "general": {
                "lr": 0.001,
                "hidden_size": 32,
                "gamma": 0.5,
                "dropout": 0.5,
            },
            "imdb-b": {"degree_feature": True},
            "imdb-m": {"degree_feature": True},
            "collab": {"degree_feature": True},
        },
    },
    "unsupervised_graph_classification": {
        "graph2vec": {
            "general": {},
            "nci1": {
                "lr": 0.001,
                "window_size": 8,
                "epoch": 10,
                "iteration": 4,
            },
            "reddit-b": {
                "lr": 0.01,
                "degree_feature": True,
                "hidden_size": 128,
            },
        }
    },
    "link_prediction": {},
    "multiplex_link_prediction": {
        "gatne": {
            "general": {},
            "twitter": {"eval_type": "1"},
        }
    },
    "multiplex_node_classification": {
        "hin2vec": {
            "general": {
                "lr": 0.025,
            },
        },
        "metapath2vec": {
            "general": {
                "walk_num": 40,
            },
        },
        "pte": {},
    },
    "heterogeneous_node_classification": {
        "gtn": {
            "general": {
                "hidden_size": 128,
                "lr": 0.005,
                "weight_decay": 0.001,
            },
        },
        "han": {
            "general": {
                "hidden_size": 128,
                "lr": 0.005,
                "weight_decay": 0.001,
            }
        },
    },
    "pretrain": {},
    "similarity_search": {
        "gcc": {
            "general": {},
        },
    },
    "attributed_graph_clustering": {},
}
