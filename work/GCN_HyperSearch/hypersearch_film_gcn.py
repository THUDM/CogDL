from cogdl import experiment

def hypersearch_gcn_all(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3, 4, 8, 10]),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu"]),
        "residual": trial.suggest_categorical("residual", [True, False]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.9, step=0.1),
        "norm": trial.suggest_categorical("norm", [None, 'layernorm', 'batchnorm']),

        "lr": trial.suggest_categorical("lr", [0.001, 0.003, 0.01, 0.03, 0.1]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-05, 1e-04]),
        "epochs": trial.suggest_categorical("epochs", [2000]),
        "patience": trial.suggest_categorical("patience", [50, 100, 200]),
    }

def hypersearch_gcn_classic(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu"]),
        "residual": trial.suggest_categorical("residual", [True, False]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.9, step=0.1),
        "norm": trial.suggest_categorical("norm", [None, 'layernorm', 'batchnorm']),

        "lr": trial.suggest_categorical("lr", [0.001, 0.003, 0.01, 0.03, 0.1]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-05, 1e-04]),
        "epochs": trial.suggest_categorical("epochs", [2000]),
        "patience": trial.suggest_categorical("patience", [200]),
    }

def hypersearch_gcn_best(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [256]),
        "num_layers": trial.suggest_categorical("num_layers", [2]),
        "activation": trial.suggest_categorical("activation", ["relu"]),
        "residual": trial.suggest_categorical("residual", [True]),
        "dropout": trial.suggest_categorical("dropout", [0.9]),
        "norm": trial.suggest_categorical("norm", [None]),

        "lr": trial.suggest_categorical("lr", [0.01]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-05]),
        "epochs": trial.suggest_categorical("epochs", [2000]),
        "patience": trial.suggest_categorical("patience", [200]),
    }

if __name__ == "__main__":
    dataset = 'film'
    model = 'gcn'
    
    if False:
        experiment(
            dataset = dataset,
            model = model,
            seed = list(range(1)),
            split = list(range(10)),
            search_space = hypersearch_gcn_classic,
            n_trials = 1,
            project = 'hypersearch',
            devices = [-1],
            # trial_log_path = './logs/trial/hypersearch_film_gcn_classic_try100.log',
            update_configs = False,
        )