from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from cogdl.utils import add_remaining_self_loops, spmm
from .base_trainer import BaseTrainer


class DAEGCTrainer(BaseTrainer):
    def __init__(self, args):
        self.num_clusters = args.num_clusters
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.T = args.T
        self.gamma = args.gamma
        self.device = args.device_id[0] if not args.cpu else "cpu"

    @staticmethod
    def build_trainer_from_args(args):
        pass

    def fit(self, model, data):
        # edge_index_2hop = model.get_2hop(data.edge_index)
        data.adj = torch.sparse_coo_tensor(
            data.edge_index, torch.ones(data.edge_index.shape[1]), torch.Size([data.x.shape[0], data.x.shape[0]])
        ).to_dense()
        data.adj += torch.eye(data.x.shape[0])
        data.apply(lambda x: x.to(self.device))
        edge_index_2hop = data.edge_index
        model = model.to(self.device)
        self.num_nodes = data.x.shape[0]

        print("Training initial embedding...")
        epoch_iter = tqdm(range(self.max_epoch))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in epoch_iter:
            model.train()
            optimizer.zero_grad()
            z = model(data.x, edge_index_2hop)
            # print(z)
            loss = model.recon_loss(z, data.adj)
            loss.backward()
            optimizer.step()
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        print("Getting cluster centers...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(
            model(data.x, data.edge_index).detach().cpu().numpy()
        )
        model.cluster_center = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, device=self.device))

        print("Self-optimizing...")
        epoch_iter = tqdm(range(self.max_epoch))
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=self.weight_decay)
        for epoch in epoch_iter:
            self.cluster_center = model.cluster_center
            # print("center=", model.cluster_center)
            model.train()
            optimizer.zero_grad()
            z = model(data.x, edge_index_2hop)
            # print("z=", z)
            Q = self.getQ(z)
            if epoch % self.T == 0:
                P = self.getP(Q).detach()
            # print(self.cluster_loss(P, Q))
            loss = model.recon_loss(z, data.adj) + self.gamma * self.cluster_loss(P, Q)
            loss.backward()
            optimizer.step()
            epoch_iter.set_description(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        return model

    def getQ(self, z):
        Q = None
        for i in range(z.shape[0]):
            dis = torch.sum((z[i].repeat(self.num_clusters, 1) - self.cluster_center) ** 2, dim=1)
            t = 1 / (1 + dis)
            t = t / torch.sum(t)
            if Q is None:
                Q = t.clone().unsqueeze(0)
            else:
                Q = torch.cat((Q, t.unsqueeze(0)), 0)
        # print("Q=", Q)
        return Q

    def getP(self, Q):
        P = torch.sum(Q, dim=0).repeat(Q.shape[0], 1)
        P = Q ** 2 / P
        P = P / (torch.ones(1, self.num_clusters, device=self.device) * torch.sum(P, dim=1).unsqueeze(-1))
        # print("P=", P)
        return P

    def cluster_loss(self, P, Q):
        # return nn.MSELoss(reduce=True, size_average=False)(P, Q)
        return nn.KLDivLoss(reduce=True, size_average=False)(P.log(), Q)

    def evaluate(self, clusters, truth):
        print("Evaluating...")
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if clusters[i] == clusters[j] and truth[i] == truth[j]:
                    TP += 1
                if clusters[i] != clusters[j] and truth[i] == truth[j]:
                    FP += 1
                if clusters[i] == clusters[j] and truth[i] != truth[j]:
                    FN += 1
                if clusters[i] != clusters[j] and truth[i] != truth[j]:
                    TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("TP", TP, "FP", FP, "TN", TN, "FN", FN)
        micro_f1 = 2 * (precision * recall) / (precision + recall)
        print(
            "Accuracy = ", precision, "NMI = ", normalized_mutual_info_score(clusters, truth), "Micro_F1 = ", micro_f1
        )
