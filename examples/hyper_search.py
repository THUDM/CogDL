import torch
import optuna

from cogdl.utils import build_args_from_dict
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.tasks import build_task


class HyperSearch(object):
    """
        This class does not need to be modified
    Args:
        Hyper-parameter search script
        func_fixed: function to obtain fixed hyper-parameters
        func_search: function to obtain hyper-parameters to search
    """
    def __init__(self, func_fixed, func_search, n_trials=30):
        self.func_fixed = func_fixed
        self.func_search = func_search
        self.dataset = None
        self.n_trials = 30

    def build_args(self, params):
        args = build_args_from_dict(params)
        if self.dataset is None:
            self.dataset = build_dataset(args)
        args.num_features = self.dataset.num_features
        args.num_classes = self.dataset.num_classes
        return args

    def objective(self, trials):
        fixed_params = self.func_fixed()
        params = self.func_search(trials)
        params.update(fixed_params)
        args = self.build_args(params)
        model = build_model(args)
        task = build_task(args, model=model, dataset=self.dataset)

        result = task.train()

        for item in list(result.keys()):
            if "Val" in item:
                result = {item: result[item]}
                break

        assert type(result) == dict
        result = list(result.values())
        return 1.0 - sum(result)/len(result)

    def final_result(self, best_params):
        params = self.func_fixed()
        params.update(best_params)
        args = self.build_args(params)
        model = build_model(args)
        task = build_task(args, model=model, dataset=self.dataset)
        result = task.train()

        for item in list(result.keys()):
            if "Val" in item:
                result.pop(item)
                break

        assert type(result) == dict
        return result

    def run(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=self.n_trials)
        best_params = study.best_params
        best_values = 1 - study.best_value

        result = self.final_result(best_params)
        return {
            "best_params": best_params,
            "best_result_in_search": best_values,
            "result": result
        }

# To tune the hyper-parameters of a given model.
# Just fill in the hyper-parameters you want to search in function `hyper_parameters_to_search`
# and the other necessary fixed hyper-parameters in function `fixed_hyper_parameters`
# Then run this script.


def hyper_parameters_to_search(trial):
    """
        Fill in hyper-parameters to search of your model
        Return hyper-parameters to search
    """
    return {
        # "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
        "lr": trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "dropout": trial.suggest_uniform("dropout", 0.3, 0.7),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
    }


def fixed_hyper_parameters():
    """
        Fill in fixed and necessary hyper-parameters of your model
        Return fixed parameters
    """
    return {
        "dataset": "cora",
        "model": "gcn",
        "patience": 100,
        "max_epoch": 500,
        "task": "node_classification",
        "cpu": not torch.cuda.is_available(),
        "seed": [0],
        "weight_decay": 5e-4
    }


def main_hypersearch(n_trials=30):
    cls = HyperSearch(fixed_hyper_parameters, hyper_parameters_to_search, n_trials=n_trials)
    result = cls.run()
    for key, val in result.items():
        print(key, val)


if __name__ == "__main__":
    main_hypersearch(40)
