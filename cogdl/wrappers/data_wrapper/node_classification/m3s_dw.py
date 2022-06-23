import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg

import torch

from .node_classification_dw import FullBatchNodeClfDataWrapper


class M3SDataWrapper(FullBatchNodeClfDataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--label-rate", type=float, default=0.2)
        parser.add_argument("--approximate", action="store_true")
        parser.add_argument("--alpha", type=float, default=0.2)
        # fmt: on

    def __init__(self, dataset, label_rate, approximate, alpha):
        super(M3SDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.label_rate = label_rate
        self.approximate = approximate
        self.alpha = alpha

    def pre_transform(self):
        data = self.dataset.data
        num_nodes = data.num_nodes
        num_classes = data.num_classes

        data.add_remaining_self_loops()
        train_nodes = torch.where(data.train_mask)[0]
        if len(train_nodes) / num_nodes > self.label_rate:
            perm = np.random.permutation(train_nodes.shape[0])
            preserve_nnz = int(num_nodes * self.label_rate)
            preserved = train_nodes[perm[:preserve_nnz]]
            masked = train_nodes[perm[preserve_nnz:]]
            data.train_mask = torch.full((data.train_mask.shape[0],), False, dtype=torch.bool)
            data.train_mask[preserved] = True
            data.test_mask[masked] = True

        # Compute absorption probability
        row, col = data.edge_index
        A = sp.coo_matrix((np.ones(row.shape[0]), (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes),).tocsr()
        D = A.sum(1).flat
        confidence = np.zeros([num_classes, num_nodes])
        confidence_ranking = np.zeros([num_classes, num_nodes], dtype=int)

        if self.approximate:
            eps = 1e-2
            for i in range(num_classes):
                q = list(torch.where(data.y == i)[0].numpy())
                q = list(filter(lambda x: data.train_mask[x], q))
                r = {idx: 1 for idx in q}
                while len(q) > 0:
                    unode = q.pop()
                    res = self.alpha / (self.alpha + D[unode]) * r[unode] if unode in r else 0
                    confidence[i][unode] += res
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
            for i in range(num_nodes):
                if data.train_mask[i]:
                    confidence[data.y[i]] += P[i]

        # Sort nodes by confidence for each class
        for i in range(num_classes):
            confidence_ranking[i] = np.argsort(-confidence[i])
        data.confidence_ranking = confidence_ranking

        self.dataset.data = data

    def pre_stage(self, stage, model_w_out):
        self.dataset.data.store("y")
        if stage > 0:
            self.dataset.data.y = model_w_out

    def post_stage(self, stage, model_w_out):
        self.dataset.data.restore("y")

    def get_dataset(self):
        return self.dataset
