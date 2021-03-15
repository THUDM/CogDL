import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cogdl.data import Dataset
from cogdl.models.supervised_model import (
    SupervisedHomogeneousNodeClassificationModel,
)
from cogdl.trainers.supervised_model_trainer import SupervisedHomogeneousNodeClassificationTrainer
from cogdl.utils.self_auxiliary_task import (
    EdgeMask,
    PairwiseDistance,
    Distance2Clusters,
    PairwiseAttrSim,
    Distance2ClustersPP,
)
from . import register_trainer
from .self_supervised_trainer import LogRegTrainer

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score

class SelfAuxiliaryTaskTrainer(SupervisedHomogeneousNodeClassificationTrainer):
    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        self.device = args.device_id[0] if not args.cpu else "cpu"
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.auxiliary_task = args.auxiliary_task
        self.hidden_size = args.hidden_size
        self.label_mask = args.label_mask
        self.sampling = args.sampling

    def resplit_data(self, data):
        trained = torch.where(data.train_mask)[0]
        perm = np.random.permutation(trained.shape[0])
        preserve_nnz = int(len(perm) * (1 - self.label_mask))
        preserved = trained[perm[:preserve_nnz]]
        masked = trained[perm[preserve_nnz:]]
        data.train_mask = torch.full((data.train_mask.shape[0],), False, dtype=torch.bool)
        data.train_mask[preserved] = True
        data.test_mask[masked] = True

    def set_agent(self):
        if self.auxiliary_task == "edgemask":
            self.agent = EdgeMask(self.data.edge_index, self.data.x, self.hidden_size, self.device)
        elif self.auxiliary_task == "pairwise-distance":
            self.agent = PairwiseDistance(self.data.edge_index, self.data.x, self.hidden_size, [(1, 2), (2, 3), (3, 5)], self.sampling, self.device)
        elif self.auxiliary_task == "distance2clusters":
            self.agent = Distance2Clusters(self.data.edge_index, self.data.x, self.hidden_size, 30, self.device)
        elif self.auxiliary_task == "pairwise-attr-sim":
            self.agent = PairwiseAttrSim(self.data.edge_index, self.data.x, self.hidden_size, 5, self.device)
        elif self.auxiliary_task == "distance2clusters++":
            self.agent = Distance2ClustersPP(
                self.data.edge_index, self.data.x, self.data.y, self.hidden_size, 5, 1, self.device
            )
        else:
            raise Exception(
                "auxiliary task must be edgemask, pairwise-distance, distance2clusters, distance2clusters++ or pairwise-attr-sim"
            )

    def _test_step(self, split="val"):
        self.data = self.original_data
        self.model.eval()
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask

        logits = self.model.predict(self.data)
        loss = F.nll_loss(logits[mask], self.data.y[mask]).item()

        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss

@register_trainer("self_auxiliary_task_pretrain")
class SelfAuxiliaryTaskPretrainer(SelfAuxiliaryTaskTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--auxiliary-task', default="none", type=str)
        parser.add_argument('--label-mask', default=0, type=float)
        parser.add_argument('--sampling', action="store_true")
        parser.add_argument("--freeze", action="store_true")
        # fmt: on
    
    def __init__(self, args):
        super().__init__(args)
        self.freeze = args.freeze

    def fit(self, model: SupervisedHomogeneousNodeClassificationModel, dataset: Dataset):
        # self.resplit_data(dataset.data)
        self.data = dataset.data
        self.original_data = dataset.data
        self.data.apply(lambda x: x.to(self.device))
        self.set_agent()
        self.model = model

        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.agent.linear.parameters()), lr=self.lr, weight_decay=self.weight_decay
        )
        self.model.to(self.device)

        self.best_model = None
        self.pretrain()
        return self.finetune()

    def pretrain(self):
        print("Pretraining")
        epoch_iter = tqdm(range(self.max_epoch))
        best_loss = np.inf
        for epoch in epoch_iter:
            if self.auxiliary_task == "distance2clusters++" and epoch % 40 == 0:
                self.agent.update_cluster()
            loss = self._pretrain_step()
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            if loss <= best_loss:
                best_loss = loss
                self.best_model = copy.deepcopy(self.model)
        self.model = copy.deepcopy(self.best_model)

    def finetune(self):
        print("Fine-tuning")
        self.original_data.apply(lambda x: x.to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.to(self.device)

        embeddings = self.best_model.get_embeddings(self.original_data.x, self.original_data.edge_index).detach()
        """
        nclass = int(torch.max(self.data.y) + 1)
        kmeans = KMeans(n_clusters=nclass, random_state=0).fit(embeddings.cpu().numpy())
        clusters = kmeans.labels_
        print("cluster NMI: %.4lf" % (normalized_mutual_info_score(clusters, self.data.y)))
        """
        if self.freeze:
            opt = {
                "idx_train": self.original_data.train_mask,
                "idx_val": self.original_data.val_mask,
                "idx_test": self.original_data.test_mask,
                "num_classes": nclass,
            }
            result = LogRegTrainer().train(embeddings, self.original_data.y, opt)
            print(f"TestAcc: {result: .4f}")
            return dict(Acc=result)
        else:
            best_score = 0
            best_loss = np.inf
            max_score = 0
            min_loss = np.inf
            epoch_iter = tqdm(range(100))
            for epoch in epoch_iter:
                self._train_step()
                train_acc, _ = self._test_step(split="train")
                val_acc, val_loss = self._test_step(split="val")
                test_acc, test_loss = self._test_step(split="test")
                epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_loss <= best_loss:  # and val_acc >= best_score:
                        best_loss = val_loss
                        best_score = val_acc
                        best_model = copy.deepcopy(self.model)
                    min_loss = np.min((min_loss, val_loss))
                    max_score = np.max((max_score, val_acc))
        return best_model

    def _pretrain_step(self):
        self.data.edge_index, self.data.x = self.agent.transform_data()
        self.model.train()
        self.optimizer.zero_grad()
        embeddings = self.model.get_embeddings(self.data.x, self.data.edge_index)
        loss = self.agent.make_loss(embeddings)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model.node_classification_loss(self.original_data)
        loss.backward()
        self.optimizer.step()

@register_trainer("self_auxiliary_task_joint")
class SelfAuxiliaryTaskJointTrainer(SelfAuxiliaryTaskTrainer):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add trainer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--auxiliary-task', default="none", type=str)
        parser.add_argument('--alpha', default=10, type=float)
        parser.add_argument('--label-mask', default=0, type=float)
        parser.add_argument('--sampling', action="store_true")
        # fmt: on

    def __init__(self, args):
        super().__init__(args)
        self.alpha = args.alpha

    def fit(self, model: SupervisedHomogeneousNodeClassificationModel, dataset: Dataset):
        # self.resplit_data(dataset.data)
        self.data = dataset.data
        self.original_data = dataset.data
        self.data.apply(lambda x: x.to(self.device))
        self.original_data.apply(lambda x: x.to(self.device))
        self.set_agent()
        self.model = model

        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.agent.linear.parameters()), lr=self.lr, weight_decay=self.weight_decay
        )
        self.model.to(self.device)
        epoch_iter = tqdm(range(self.max_epoch))

        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf

        for epoch in epoch_iter:
            if self.auxiliary_task == "distance2clusters++" and epoch % 40 == 0:
                self.agent.update_cluster()
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            test_acc, test_loss = self._test_step(split="test")
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
        print(f"Valid accurracy = {best_score}")

        return best_model

    def _train_step(self):
        self.data.edge_index, self.data.x = self.agent.transform_data()
        self.model.train()
        self.optimizer.zero_grad()
        embeddings = self.model.get_embeddings(self.data.x, self.data.edge_index)
        loss = self.model.node_classification_loss(self.data) + self.alpha * self.agent.make_loss(embeddings)
        loss.backward()
        self.optimizer.step()

