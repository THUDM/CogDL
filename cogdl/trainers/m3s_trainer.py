from tqdm import tqdm
import copy

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
from sklearn.cluster import KMeans

import torch
from .base_trainer import BaseTrainer


class M3STrainer(BaseTrainer):
    def __init__(self, args):
        super(M3STrainer, self).__init__()
        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        self.epochs = args.epochs_per_stage
        self.num_classes = args.num_classes
        self.hidden_size = args.hidden_size
        self.weight_decay = args.weight_decay
        self.num_clusters = args.num_clusters
        self.num_stages = args.num_stages
        self.label_rate = args.label_rate
        self.num_new_labels = args.num_new_labels
        self.approximate = args.approximate
        self.lr = args.lr
        self.alpha = args.alpha

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def preprocess(self, data):
        data.add_remaining_self_loops()
        train_nodes = torch.where(self.data.train_mask)[0]
        if len(train_nodes) / self.num_nodes > self.label_rate:
            perm = np.random.permutation(train_nodes.shape[0])
            preserve_nnz = int(self.num_nodes * self.label_rate)
            preserved = train_nodes[perm[:preserve_nnz]]
            masked = train_nodes[perm[preserve_nnz:]]
            data.train_mask = torch.full((data.train_mask.shape[0],), False, dtype=torch.bool)
            data.train_mask[preserved] = True
            data.test_mask[masked] = True

        # Compute absorption probability
        row, col = data.edge_index
        A = sp.coo_matrix(
            (np.ones(row.shape[0]), (row.numpy(), col.numpy())),
            shape=(self.num_nodes, self.num_nodes),
        ).tocsr()
        D = A.sum(1).flat
        self.confidence = np.zeros([self.num_classes, self.num_nodes])
        self.confidence_ranking = np.zeros([self.num_classes, self.num_nodes], dtype=int)

        if self.approximate:
            eps = 1e-2
            for i in range(self.num_classes):
                q = list(torch.where(data.y == i)[0].numpy())
                q = list(filter(lambda x: data.train_mask[x], q))
                r = {idx: 1 for idx in q}
                while len(q) > 0:
                    unode = q.pop()
                    res = self.alpha / (self.alpha + D[unode]) * r[unode] if unode in r else 0
                    self.confidence[i][unode] += res
                    r[unode] = 0
                    for vnode in A.indices[A.indptr[unode] : A.indptr[unode + 1]]:
                        val = res / self.alpha
                        if vnode in r:
                            r[vnode] += val
                        else:
                            r[vnode] = val
                        # print(vnode, val)
                        if val > eps * D[vnode] and vnode not in q:
                            q.append(vnode)
        else:
            L = sp.diags(D, dtype=np.float32) - A
            L += self.alpha * sp.eye(L.shape[0], dtype=L.dtype)
            P = slinalg.inv(L.tocsc()).toarray().transpose()
            for i in range(self.num_nodes):
                if data.train_mask[i]:
                    self.confidence[data.y[i]] += P[i]

        # Sort nodes by confidence for each class
        for i in range(self.num_classes):
            self.confidence_ranking[i] = np.argsort(-self.confidence[i])
            print(self.confidence_ranking[i][:10])
        return data

    def fit(self, model, dataset):
        self.data = dataset[0]
        self.num_nodes = self.data.x.shape[0]
        self.data = self.preprocess(self.data)
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = dataset.get_loss_fn()
        self.evaluator = dataset.get_evaluator()

        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf

        print("Training on original split...")
        self.data = self.data.to(self.device)
        self.model = self.model.to(self.device)
        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
            if val_loss <= min_loss or val_acc >= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss.cpu()))
                max_score = np.max((max_score, val_acc))

        with self.data.local_graph():
            for stage in range(self.num_stages):
                print(f"Stage # {stage}:")
                emb = best_model.get_embeddings(self.data)
                # self.data = self.data.apply(lambda x: x.cpu())
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(emb)
                clusters = kmeans.labels_

                # Compute centroids μ_m of each class m in labeled data and v_l of each cluster l in unlabeled data.
                labeled_centroid = np.zeros([self.num_classes, self.hidden_size])
                unlabeled_centroid = np.zeros([self.num_clusters, self.hidden_size])
                for i in range(self.num_nodes):
                    if self.data.train_mask[i]:
                        labeled_centroid[self.data.y[i]] += emb[i]
                    else:
                        unlabeled_centroid[clusters[i]] += emb[i]

                # Align labels for each cluster
                align = np.zeros(self.num_clusters, dtype=int)
                for i in range(self.num_clusters):
                    for j in range(self.num_classes):
                        if np.linalg.norm(unlabeled_centroid[i] - labeled_centroid[j]) < np.linalg.norm(
                            unlabeled_centroid[i] - labeled_centroid[align[i]]
                        ):
                            align[i] = j

                # Add new labels
                for i in range(self.num_classes):
                    t = self.num_new_labels
                    for j in range(self.num_nodes):
                        idx = self.confidence_ranking[i][j]
                        if not self.data.train_mask[idx]:
                            if t <= 0:
                                break
                            t -= 1
                            if align[clusters[idx]] == i:
                                self.data.train_mask[idx] = True
                                self.data.y[idx] = i

                # Training
                self.data = self.data.to(self.device)
                epoch_iter = tqdm(range(self.epochs))
                for epoch in epoch_iter:
                    self._train_step()
                    train_acc, _ = self._test_step(split="train")
                    val_acc, val_loss = self._test_step(split="val")
                    epoch_iter.set_description(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                    if val_loss <= min_loss or val_acc >= max_score:
                        if val_loss <= best_loss:  # and val_acc >= best_score:
                            best_loss = val_loss
                            best_score = val_acc
                            best_model = copy.deepcopy(self.model)
                        min_loss = np.min((min_loss, val_loss.cpu()))
                        max_score = np.max((max_score, val_acc))
        print("Val accuracy %.4lf" % (best_score))

        return best_model

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.node_classification_loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.predict(self.data)
        if split == "train":
            mask = self.data.train_mask
        elif split == "val":
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask

        loss = self.loss_fn(logits[mask], self.data.y[mask])
        metric = self.evaluator(logits[mask], self.data.y[mask])
        return metric, loss
