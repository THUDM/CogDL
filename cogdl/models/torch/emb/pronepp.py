import optuna
import numpy as np
import random

from .. import BaseModel
from cogdl.utils.prone_utils import propagate, get_embedding_dense


class PlainFilter(object):
    def __init__(self, filter_types, svd):
        self.filter_types = filter_types
        self.svd = svd
        # load adjacency matrix and raw embedding

    def __call__(self, emb, adj):
        if len(self.filter_types) == 1 and self.filter_types[0] == "identity":
            return emb

        dim = emb.shape[1]
        prop_result = []
        for tp in self.filter_types:
            prop_result.append(propagate(adj, emb, tp))
        prop_result_emb = np.concatenate(prop_result, axis=1)
        if self.svd:
            prop_result_emb = get_embedding_dense(prop_result_emb, dim)
        return prop_result_emb


class Search(object):
    def __init__(self, filter_types, max_evals, svd, loss_type, n_workers):
        self.prop_types = filter_types
        self.max_evals = max_evals
        self.svd = svd
        self.loss_type = loss_type
        self.n_workers = n_workers

        # load adjacency matrix and raw embedding
        self.num_edges = self.num_nodes = self.dim = 0
        self.laplacian = None
        self.emb = self.adj = None

        self.batch_size = 64

    def build_search_space(self, trial):
        space = {}
        for f in self.prop_types:
            space[f] = trial.suggest_categorical(f, [0, 1])
        if space.get("heat", 0) == 1:
            space["t"] = trial.suggest_uniform("t", 0.1, 0.9)
        if space.get("gaussian", 0) == 1:
            space["mu"] = trial.suggest_uniform("mu", 0.1, 2)
            space["theta"] = trial.suggest_uniform("theta", 0.2, 1.5)
        if space.get("ppr", 0) == 1:
            space["alpha"] = trial.suggest_uniform("alpha", 0.2, 0.8)
        return space

    def init_data(self, emb, adj):
        self.num_nodes, self.dim = emb.shape
        self.num_edges = adj.nnz
        self.emb = emb
        self.adj = adj

        if self.loss_type == "infonce":
            neg_index = []
            for i in range(self.num_nodes):
                select = np.random.choice(self.num_nodes, self.batch_size, replace=False)
                while i in select:
                    select = np.random.choice(self.num_nodes, self.batch_size, replace=False)
                neg_index.append(select)
            self.neg_index = np.array(neg_index)
            self.neg_emb = self.emb[self.neg_index]
        elif self.loss_type == "infomax":
            # self.permutation = np.random.permutation(np.arange(self.num_nodes))
            pass

    def prop(self, params):
        prop_types = [key for key, value in params.items() if value == 1 and key in self.prop_types]
        if not prop_types:
            print(" -- dropped -- ")
            return None, None

        prop_result_list = []
        for selected_prop in prop_types:
            prop_result = propagate(self.adj, self.emb, selected_prop, params)
            prop_result_list.append(prop_result)

        if self.loss_type == "infomax":
            neg_prop_result = []
            pmt = self.permutation
            for s_prop in prop_types:
                neg_prop = propagate(self.adj, self.emb[pmt], s_prop, params)
                neg_prop[pmt] = neg_prop
                neg_prop_result.append(neg_prop)
            return np.array(prop_result_list), np.array(neg_prop_result)
        elif self.loss_type == "infonce":
            return np.array(prop_result_list), None
        elif self.loss_type == "sparse":
            return np.array(prop_result_list), None
        else:
            raise ValueError("use 'infonce', 'infomax' or 'sparse' loss, currently using {}".format(self.loss_type))

    def target_func(self, trial):
        params = self.build_search_space(trial)
        self.permutation = np.random.permutation(np.arange(self.num_nodes))
        prop_result_emb, neg_prop_result_emb = self.prop(params)
        if prop_result_emb is None:
            return 100
        if self.loss_type == "infomax":
            loss = self.infomax_loss(prop_result_emb, neg_prop_result_emb)
        elif self.loss_type == "infonce":
            loss = self.infonce_loss(prop_result_emb)
        else:
            raise ValueError("loss type must be in ['infomax', 'infonce']")
        return loss

    def infonce_loss(self, prop_emb_list, *args, **kwargs):
        T = 0.07

        pos_infos = []
        for smoothed in prop_emb_list:
            pos_info = np.exp(np.sum(smoothed * self.emb, -1) / T)
            assert pos_info.shape == (self.num_nodes,)
            pos_infos.append(pos_info)

        neg_infos = []
        for idx, smoothed in enumerate(prop_emb_list):
            neg_info = np.exp(
                np.sum(np.tile(smoothed[:, np.newaxis, :], (1, self.batch_size, 1)) * self.neg_emb, -1) / T
            ).sum(-1)
            assert neg_info.shape == (self.num_nodes,)
            neg_infos.append(neg_info + pos_infos[idx])

        pos_neg = np.array(pos_infos) / np.array(neg_infos)
        if np.isnan(pos_neg).any():
            pos_neg = np.nan_to_num(pos_neg)

        loss = -np.log(pos_neg).mean()
        return loss / 10

    def infomax_loss(self, prop_emb_list, neg_prop_emb_list):
        prop_result = np.concatenate(prop_emb_list, axis=1)
        if self.svd:
            prop_result = get_embedding_dense(prop_result, self.dim)

        def sigmoid(x):
            return 1.0 / (1 + np.exp(-x))

        pos_glb = prop_result.mean(0)
        pos_info = sigmoid(pos_glb.dot(prop_result.T))
        pos_loss = np.mean(np.log(pos_info)).mean()

        neg_loss = 0
        neg_step = 1
        for _ in range(neg_step):
            neg_prop_result = np.concatenate(neg_prop_emb_list, axis=1)
            if self.svd:
                neg_prop_result = get_embedding_dense(neg_prop_result, self.dim)

            neg_info = sigmoid(pos_glb.dot(neg_prop_result.T))
            neg_loss += np.mean(np.log(1 - neg_info)).mean()
            random.shuffle(neg_prop_emb_list)

        return -(pos_loss + neg_loss) / (1 + neg_step)

    def __call__(self, emb, adj):
        self.init_data(emb, adj)
        study = optuna.create_study()
        study.optimize(self.target_func, n_jobs=self.n_workers, n_trials=self.max_evals)
        best_params = study.best_params

        best_result = self.prop(best_params)[0]
        best_result = np.concatenate(best_result, axis=1)
        print(f"best parameters: {best_params}")

        if self.svd:
            best_result = get_embedding_dense(best_result, self.dim)
        return best_result


class ProNEPP(BaseModel):
    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            filter_types=args.filter_types,
            max_evals=args.max_evals,
            loss_type=args.loss,
            svd=not args.no_svd,
            n_workers=args.num_workers,
            search=not args.no_search,
        )

    def __init__(self, filter_types, svd, search, max_evals=None, loss_type=None, n_workers=None):
        super(ProNEPP, self).__init__()
        if search:
            self.model = Search(filter_types, max_evals, svd, loss_type, n_workers)
        else:
            self.model = PlainFilter(filter_types, svd)

    def __call__(self, emb, adj):
        enhanced_emb = self.model(emb, adj)
        return enhanced_emb
