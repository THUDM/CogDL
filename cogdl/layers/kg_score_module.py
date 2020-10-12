import torch
from torch import Tensor
import torch.nn.functional as F

class BaseScorer(torch.nn.Module):
    def score(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, combine: str) -> Tensor:
        """To score a specific tuple or a set of tuples with multiple relations, subjectives or objectives.
           combine can be "spo", "sp_", "_po" or "s_p" to denote the different type of scoring one wants to perform.
        """
        n = p_emb.size(0)

        if combine == "spo":
            assert s_emb.size(0) == n and o_emb.size(0) == n
            out = self.score_tuple(s_emb, p_emb, o_emb)
        elif combine == "sp_":
            assert s_emb.size(0) == n
            n_o = o_emb.size(0)
            s_embs = s_emb.repeat_interleave(n_o, 0)
            p_embs = p_emb.repeat_interleave(n_o, 0)
            o_embs = o_emb.repeat((n, 1))
            out = self.score_tuple(s_embs, p_embs, o_embs)
        elif combine == "_po":
            assert o_emb.size(0) == n
            n_s = s_emb.size(0)
            s_embs = s_emb.repeat((n, 1))
            p_embs = p_emb.repeat_interleave(n_s, 0)
            o_embs = o_emb.repeat_interleave(n_s, 0)
            out = self.score_tuple(s_embs, p_embs, o_embs)
        elif combine == "s_o":
            n = s_emb.size(0)
            assert o_emb.size(0) == n
            n_p = p_emb.size(0)
            s_embs = s_emb.repeat_interleave(n_p, 0)
            p_embs = p_emb.repeat((n, 1))
            o_embs = o_emb.repeat_interleave(n_p, 0)
            out = self.score_tuple(s_embs, p_embs, o_embs)
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(n, -1)

    def score_tuple(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor) -> Tensor:
        """The score function for a specific tuple"""
        raise NotImplementedError(
            "Score function hasn't been implemented for this scorer"
        )

class DistMultScorer(BaseScorer):
    
    def score(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        if combine == "spo":
            out = (s_emb * p_emb * o_emb).sum(dim=1)
        elif combine == "sp_":
            out = (s_emb * p_emb).mm(o_emb.transpose(0, 1))
        elif combine == "_po":
            out = (o_emb * p_emb).mm(s_emb.transpose(0, 1))
        else:
            raise ValueError('cannot handle combine="{}".format(combine) in DistMult')

        return out.view(n, -1)

class TransEScorer(BaseScorer):

    def __init__(self, norm = 1.0):
        super(TransEScorer, self).__init__()
        self._norm = norm

    def score(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        if combine == "spo":
            out = -F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "sp_":
            out = -torch.cdist(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "_po":
            out = -torch.cdist(o_emb - p_emb, s_emb, p=self._norm)
        else:
            return ValueError('cannot handle combine="{}".format(combine) in TransE')
        return out.view(n, -1)


