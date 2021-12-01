BEST_CONFIGS = {
    "chebyshev": {"general": {}},
    "dropedge_gcn": {"general": {}},
    "gat": {
        "general": {"lr": 0.005, "epochs": 1000},
        "citeseer": {"weight_decay": 0.001},
        "pubmed": {"weight_decay": 0.001},
        "ppi-large": {
            "nhead": 4,
            "hidden_size": 256,
            "lr": 0.005,
            "residual": True,
            "num_layers": 3,
            "weight_decay": 0.0,
            "attn_drop": 0,
            "dropout": 0,
        },
    },
    "gcn": {
        "general": {},
        "ppi-large": {
            "lr": 0.005,
            "hidden_size": 2048,
            "num_layers": 2,
            "dropout": 0.3,
            "epochs": 7000,
            # 78.04
        },
        "flickr": {
            "lr": 0.01,
            "dropout": 0.5,
            "weight_decay": 0.0,
            "hidden_size": 256,
        },
    },
    "sage": {
        "general": {},
    },
    "gcnii": {
        "general": {
            "epochs": 1000,
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
            "epochs": 2000,
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
        "reddit": {
            "num_layers": 3,
            "hidden_size": 256,
            "alpha": 0.2,
            "wd1": 0,
            "wd2": 0,
            "lr": 0.001,
            "residual": True,
            "epochs": 1000,
            "lmbda": 1.0,
            # 96.42
        },
        "flickr": {
            "lr": 0.002,
            "num_layers": 4,
            "residual": True,
            "wd1": 0.0005,
            "wd2": 0.0005,
            "lmbda": 1.0,
            "alpha": 0.1,
        }
        # 52.54
    },
    "gdc_gcn": {
        "general": {"hidden_size": 16},
    },
    "grand": {
        "general": {
            "epochs": 1000,
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
            "epochs": 1000,
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
            "epochs": 1000,
        },
        "cora": {"dropout": 0.6},
        "citeseer": {"dropout": 0.6},
    },
    "unet": {
        "general": {
            "epochs": 1000,
            "n_dropout": 0.90,
            "adj_dropout": 0.05,
            "hidden_size": 128,
            "aug_adj": False,
            "improved": False,
            "n_pool": 4,
            "pool_rate": [0.7, 0.5, 0.5, 0.4],
        },
    },
    "ppnp": {
        "general": {},
        "flickr": {
            "lr": 0.005,
            "weight_decay": 0.001,
            "dropout": 0.3,
            "num_layers": 2,
            "hidden_size": 256,
            "epochs": 1000,
            "num_iterations": 4,
            "alpha": 0.5,
            # 0.5227
        },
        "cora": {
            "lr": 0.01,
            "weight_decay": 0.000887,
            "alpha": 0.1087,
            "num_iterations": 7,
            "num_layers": 2,
            "epochs": 1000,
            "hidden_size": 32,
            "dropout": 0.8185,
            #
        },
        "citeseer": {
            "num_iterations": 4,
            "hidden_size": 128,
            "lr": 0.01,
            "dropout": 0.3095,
            "alpha": 0.171,
            "weight_decay": 0.0004117,
            "num_layers": 2,
            "epochs": 1000,
            # 0.7199
        },
        "pubmed": {
            "num_iterations": 10,
            "hidden_size": 256,
            "dropout": 0.3766,
            "alpha": 0.1201,
            "weight_decay": 0.0002754,
            "epochs": 1000,
            "lr": 0.001,
        },
        "reddit": {
            "lr": 0.005,
            "hidden_size": 256,
            "dropout": 0.3751,
            "alpha": 0.1995,
            "num_layers": 3,
            "weight_decay": 5.918e-6,
            "num_iterations": 3,
            # 96.26
        },
    },
    "correct_smooth_mlp": {
        "general": {},
        "ogbn_arxiv": {
            "correct_norm": "row",
            "smooth_norm": "col",
            "correct_alpha": 0.9791632871592579,
            "smooth_alpha": 0.7564990804200602,
            "num_correct_prop": 50,
            "num_smooth_prop": 50,
            "autoscale": True,
            "norm": "batchnorm",
        },
        "ogbn_products": {
            "correct_norm": "sym",
            "smooth_norm": "row",
            "correct_alpha": 1.0,
            "smooth_alpha": 0.8,
            "num_correct_prop": 50,
            "num_smooth_prop": 50,
            "autoscale": False,
            "scale": 10.0,
            "norm": "batchnorm",
            "act_first": True,
        },
    },
    "sagn": {
        "general": {
            "data_gpu": True,
            "lr": 0.001,
            "hidden-size": 512,
            "attn-drop": 0.0,
            "dropout": 0.7,
        },
        "flickr": {
            "threshold": 0.5,
            "label-hop": 2,
            "weight-decay": 3e-6,
            "nstage": [50, 50, 50],
            "nhop": 2,
            "batch-size": 256,
        },
        "reddit": {
            "threshold": 0.9,
            "lr": 0.0001,
            "batch-size": 1000,
            "nhop": 2,
            "label-nhop": 4,
            "weight-decay": 0.0,
            "nstage": [500, 500, 500],
        },
    },
    "deepwalk": {
        "general": {},
    },
    "dngr": {
        "general": {
            "hidden_size": 128,
            "lr": 0.001,
            "epochs": 500,
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
        "general": {"window_size": 10, "num_round": 1000},
        "wikipedia": {"window_size": 1},
        "blogcatalog": {"num_round": 10000},
    },
    "node2vec": {
        "general": {},
    },
    "prone": {
        "general": {"step": 10},
        "ppi-ne": {"mu": 0.0},
        "wikipedia": {"mu": -4.0},
        "dblp-ne": {"mu": -1.2, "theta": 2.0},
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
            "epochs": 1000,
            "patience": 20,
        },
        "cora": {
            "lr": 0.0005,
            "weight_decay": 0.00001,
            "tau": 0.4,
            "drop_feature_rates": [0.3, 0.4],
            "drop_edge_rates": [0.2, 0.4],
            "epochs": 200,
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
        "epochs": 3000,
    },
    "revgcn": {
        "general": {},
        "cora": {
            "hidden_size": 128,
            "lr": 0.001,
            "dropout": 0.4706458854,
            "weight_decay": 0.0008907,
            "norm": "layernorm",
            "num_layers": 10,
            # 81.40
        },
    },
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
            "epochs": 20,
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
    "graph2vec": {
        "general": {},
        "nci1": {
            "lr": 0.001,
            "window_size": 8,
            "epochs": 10,
            "iteration": 4,
        },
        "reddit-b": {
            "lr": 0.01,
            "degree_feature": True,
            "hidden_size": 128,
        },
    },
    "gatne": {
        "general": {},
        "twitter": {"eval_type": "1"},
    },
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
}
