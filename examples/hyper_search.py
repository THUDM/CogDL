import torch
import optuna

from cogdl.utils import build_args_from_dict
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.tasks import build_task
from cogdl.utils import set_random_seed


N_SEED = 5


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
        self.n_trials = n_trials

    def build_args(self, params):
        args = build_args_from_dict(params)
        if self.dataset is None:
            self.dataset = build_dataset(args)
        args.num_features = self.dataset.num_features
        args.num_classes = self.dataset.num_classes
        return args

    def run_n_seed(self, args):
        result_list = []
        for seed in range(N_SEED):
            set_random_seed(seed)

            model = build_model(args)
            task = build_task(args, model=model, dataset=self.dataset)

            result = task.train()
            result_list.append(result)
        return result_list

    def objective(self, trials):
        fixed_params = self.func_fixed()
        params = self.func_search(trials)
        params.update(fixed_params)
        args = self.build_args(params)

        result_list = self.run_n_seed(args)

        item = result_list[0]
        key = None
        for _key in item.keys():
            if "Val" in _key:
                key = _key
                break
        if not key:
            raise KeyError
        val_meansure = [x[key] for x in result_list]
        mean = sum(val_meansure) / len(val_meansure)

        return 1.0 - mean

    def final_result(self, best_params):
        params = self.func_fixed()
        params.update(best_params)
        args = self.build_args(params)

        result_list = self.run_n_seed(args)
        keys = list(result_list[0].keys())
        result = {}
        for key in keys:
            result[key] = sum(x[key] for x in result_list) / len(result_list)

        return result

    def run(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=10)
        best_params = study.best_params
        best_values = 1 - study.best_value

        result = self.final_result(best_params)
        return {"best_params": best_params, "best_result_in_search": best_values, "result": result}


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
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "n_dropout": trial.suggest_uniform("n_dropout", 0.5, 0.92),
        "adj_dropout": trial.suggest_uniform("adj_dropout", 0.0, 0.3),
        "aug_adj": trial.suggest_categorical("aug_adj", [True, False]),
        "improved": trial.suggest_categorical("improved", [True, False]),
        # "activation": trial.suggest_categorical("activation", ["relu", "identity"]),
        # "num_layers": trial.suggest_int("num_layers", 1, 3),
    }


def fixed_hyper_parameters():
    """
    Fill in fixed and necessary hyper-parameters of your model
    Return fixed parameters
    """
    cpu = not torch.cuda.is_available()
    return {
        "weight_decay": 0.001,
        "max_epoch": 1000,
        "patience": 100,
        "cpu": cpu,
        "device_id": [0],
        "seed": [0, 1, 2],
        "task": "node_classification",
        "model": "unet",
        "dataset": "cora",
        "n_pool": 4,
        "pool_rate": [0.7, 0.5, 0.5, 0.4],
        "missing_rate": -1,
        "activation": "identity",
    }


def main_hypersearch(n_trials=50):
    cls = HyperSearch(fixed_hyper_parameters, hyper_parameters_to_search, n_trials=n_trials)
    result = cls.run()
    for key, val in result.items():
        print(key, val)


if __name__ == "__main__":
    main_hypersearch(50)
