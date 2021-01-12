from abc import ABC, abstractmethod
from typing import Any
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cogdl.data import Dataset
from cogdl.models.supervised_model import (
    SupervisedHeterogeneousNodeClassificationModel,
    SupervisedHomogeneousNodeClassificationModel,
)
from cogdl.trainers.supervised_trainer import SupervisedHomogeneousNodeClassificationTrainer
from cogdl.trainers.self_task import EdgeMask, PairwiseDistance, Distance2Clusters, PairwiseAttrSim, Distance2ClustersPP


class SelfTaskTrainer(SupervisedHomogeneousNodeClassificationTrainer):
    def __init__(self, args):
        self.device = args.device_id[0] if not args.cpu else "cpu"
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.auxiliary_task = args.auxiliary_task
        self.hidden_size = args.hidden_size
        self.alpha = args.alpha
        self.label_mask = args.label_mask

    @staticmethod
    def build_trainer_from_args(args):
        pass

    def resplit_data(self, data):
        trained = torch.where(data.train_mask)[0]
        perm = np.random.permutation(trained.shape[0])
        preserve_nnz = int(len(perm) * (1 - self.label_mask))
        preserved = trained[perm[:preserve_nnz]]
        masked = trained[perm[preserve_nnz:]]
        data.train_mask = torch.zeros(data.train_mask.shape[0], dtype=bool)
        data.train_mask[preserved] = True
        data.test_mask[masked] = True

    def fit(self, model: SupervisedHeterogeneousNodeClassificationModel, dataset: Dataset):
        # self.resplit_data(dataset.data)
        self.data = dataset.data
        self.original_data = dataset.data
        self.data.apply(lambda x: x.to(self.device))
        self.original_data.apply(lambda x: x.to(self.device))
        if self.auxiliary_task == "edgemask":
            self.agent = EdgeMask(self.data.edge_index, self.data.x, self.hidden_size, self.device)
        elif self.auxiliary_task == "pairwise-distance":
            self.agent = PairwiseDistance(self.data.edge_index, self.data.x, self.hidden_size, 3, self.device)
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
            elif self.auxiliary_task == "scalable-distance-pred" and epoch % 10 == 0:
                self.agent.update()
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            test_acc, test_loss = self._test_step(split="test")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )
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
