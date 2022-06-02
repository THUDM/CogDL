from cogdl import experiment
import json

def search_space(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "dropout": trial.suggest_uniform("dropout", 0.5, 0.8),
    }

result, log = experiment(dataset="cora", model="gcn", seed=[1, 2, 3], search_space=search_space, n_trials=10, trial_log_path=None)

for key in log:
    value = log[key]
    print ("\n", "-----"*10, key, "-----"*10, "\n",)
    print (json.dumps(value, indent=4, ensure_ascii=False))

