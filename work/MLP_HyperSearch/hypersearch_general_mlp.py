from cogdl import experiment

def hypersearch_mlp_all(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3, 4, 8, 10]),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.9, step=0.1),
        "norm": trial.suggest_categorical("norm", [None, 'layernorm', 'batchnorm']),

        "lr": trial.suggest_categorical("lr", [0.001, 0.003, 0.01, 0.03, 0.1]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-05, 1e-04]),
        "epochs": trial.suggest_categorical("epochs", [2000]),
        "patience": trial.suggest_categorical("patience", [50, 100, 200]),
    }

def hypersearch_mlp_classic(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.9, step=0.1),
        "norm": trial.suggest_categorical("norm", [None, 'layernorm', 'batchnorm']),

        "lr": trial.suggest_categorical("lr", [0.001, 0.003, 0.01, 0.03, 0.1]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-05, 1e-04]),
        "epochs": trial.suggest_categorical("epochs", [2000]),
        "patience": trial.suggest_categorical("patience", [200]),
    }

def hypersearch_mlp_best(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [256]),
        "num_layers": trial.suggest_categorical("num_layers", [2]),
        "activation": trial.suggest_categorical("activation", ["relu"]),
        "dropout": trial.suggest_categorical("dropout", [0.9]),
        "norm": trial.suggest_categorical("norm", [None]),

        "lr": trial.suggest_categorical("lr", [0.01]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-05]),
        "epochs": trial.suggest_categorical("epochs", [2000]),
        "patience": trial.suggest_categorical("patience", [200]),
    }

if __name__ == "__main__":
    dataset, model, typ = '', 'mlp', 0

    seed, split = list(range(5)), list(range(1))
    if dataset in ["cora_geom", "citeseer_geom", "pubmed_geom", "chameleon", "squirrel", "film", "cornell", "texas", "wisconsin"]:
        split = list(range(10))
    
    if typ == 0:
        seed, split = [0, 1, 2], [0]
        search_space = hypersearch_mlp_best
        n_trials = 1
        trial_log_path = None
        update_configs = False
    
    elif typ == 1:
        search_space = hypersearch_mlp_classic
        n_trials = 100
        trial_log_path = './logs/trial/hypersearch_%s_%s_classic_try%d.log' % (dataset, model, n_trials)
        update_configs = True
    
    elif typ == 2:
        search_space = hypersearch_mlp_classic
        n_trials = 300
        trial_log_path = './logs/trial/hypersearch_%s_%s_classic_try%d.log' % (dataset, model, n_trials)
        update_configs = True
    
    elif typ == 3:
        search_space = hypersearch_mlp_all
        n_trials = 100
        trial_log_path = './logs/trial/hypersearch_%s_%s_all_try%d.log' % (dataset, model, n_trials)
        update_configs = True
    
    elif typ == 4:
        search_space = hypersearch_mlp_all
        n_trials = 300
        trial_log_path = './logs/trial/hypersearch_%s_%s_all_try%d.log' % (dataset, model, n_trials)
        update_configs = True

    if True:
        experiment(
            dataset = dataset,
            model = model,
            seed = seed,
            split = split,
            search_space = search_space,
            n_trials = n_trials,
            project = 'hypersearch',
            devices = [-1],
            trial_log_path = trial_log_path,
            update_configs = update_configs,
        )