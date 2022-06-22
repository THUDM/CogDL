import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..base import ModificationAttack
from cogdl.utils.grb_utils import feat_preprocess, adj_preprocess, getGraph, getGRBGraph
from cogdl.data import Graph


class FGA(ModificationAttack):
    """
    FGA: Fast Gradient Attack on Network Embedding (https://arxiv.org/pdf/1809.02797.pdf)
    """

    def __init__(self, n_edge_mod, loss=F.cross_entropy, allow_isolate=True, device="cpu", verbose=True):
        self.n_edge_mod = n_edge_mod
        self.allow_isolate = allow_isolate
        self.loss = loss
        self.device = device
        self.verbose = verbose

    def attack(self, model, graph: Graph, feat_norm=None, adj_norm_func=None):

        adj, features = getGRBGraph(graph)
        features = feat_preprocess(features=features, feat_norm=feat_norm, device=self.device)
        adj_tensor = adj_preprocess(adj=adj, adj_norm_func=adj_norm_func, device=self.device)
        model.to(self.device)
        pred_origin = model(getGraph(adj_tensor, features, device=self.device))
        labels_origin = torch.argmax(pred_origin, dim=1)

        adj_attack = self.modification(
            model=model,
            adj_origin=adj,
            features_origin=features,
            labels_origin=labels_origin,
            index_target=graph.test_nid,
            feat_norm=feat_norm,
            adj_norm_func=adj_norm_func,
        )

        return getGraph(adj_attack, graph.x, graph.y, device=self.device)

    def modification(
        self, model, adj_origin, features_origin, labels_origin, index_target, feat_norm=None, adj_norm_func=None
    ):
        model.eval()
        if type(adj_origin) == torch.Tensor:
            adj_attack = adj_origin.clone().to_dense()
        else:
            adj_attack = adj_origin.todense()
            adj_attack = torch.FloatTensor(adj_attack)
        features_origin = feat_preprocess(features=features_origin, feat_norm=feat_norm, device=self.device)
        adj_attack.requires_grad = True
        n_edge_flip = 0
        for _ in tqdm(range(adj_attack.shape[1])):
            if n_edge_flip >= self.n_edge_mod:
                break
            adj_attack_tensor = adj_preprocess(adj=adj_attack, adj_norm_func=adj_norm_func, device=self.device)
            # print(type(adj_attack_tensor), adj_attack_tensor.is_sparse)
            # degs = torch.sparse.sum(adj_attack_tensor, dim=1)
            degs = adj_attack_tensor.sum(dim=1)
            # pred = model(getGraph(sp.csr_matrix(adj_attack.detach().cpu().numpy()), features_origin))
            pred = model(getGraph(adj_attack_tensor, features_origin, device=self.device))
            loss = self.loss(pred[index_target], labels_origin[index_target])
            grad = torch.autograd.grad(loss, adj_attack)[0]
            grad = (grad + grad.T) / torch.Tensor([2.0]).to(self.device)
            grad_max = torch.max(grad[index_target], dim=1)
            index_max_i = torch.argmax(grad_max.values)
            index_max_j = grad_max.indices[index_max_i]
            index_max_i = index_target[index_max_i]
            if adj_attack[index_max_i][index_max_j] == 0:
                adj_attack.data[index_max_i][index_max_j] = 1
                adj_attack.data[index_max_j][index_max_i] = 1
                n_edge_flip += 1
            else:
                if self.allow_isolate:
                    adj_attack.data[index_max_i][index_max_j] = 0
                    adj_attack.data[index_max_j][index_max_i] = 0
                    n_edge_flip += 1
                else:
                    if degs[index_max_i] > 1 and degs[index_max_j] > 1:
                        adj_attack.data[index_max_i][index_max_j] = 0
                        adj_attack.data[index_max_j][index_max_i] = 0
                        degs[index_max_i] -= 1
                        degs[index_max_j] -= 1
                        n_edge_flip += 1

        adj_attack = adj_attack.detach().cpu().numpy()
        adj_attack = sp.csr_matrix(adj_attack)
        if self.verbose:
            print("FGA attack finished. {:d} edges were flipped.".format(n_edge_flip))

        return adj_attack
